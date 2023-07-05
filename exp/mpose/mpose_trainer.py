from pathlib import Path
import torch
from typing import Union
from copy import copy, deepcopy

import timm
import numpy as np
from utils import LOGGER
from utils.torch_utils import (
    torch_distributed_zero_first,
    de_parallel,
    build_dataloader,
    ModelWrapper,
)
from engine.trainer import BaseTrainer
from utils import colorstr
from data_tools.graph_tools.graph import Graph
from data_tools.bop_dataset import BOPDataset
from data_tools.dummy_dataset import DummyDataset
from data_tools.dataset import DatasetLM
from utils.flags import Mode

# import hiera
from models.backbones.hiera import hiera_base_224

from gat_inf import GraphNet, GraphNetv2, AttentionMode

from exp.mpose.mpose_validator import MposeValidator

from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing
from pytorch3d.structures import Meshes
import torch

class MposeTrainer(BaseTrainer):
    def get_dataset(
        self, img_path, mode=Mode.TRAIN, use_cache=True, single_object=False
    ):
        return BOPDataset(img_path, mode, use_cache=True, single_object=False)
    
    def postprocess_batch(self, pred, batch):
        return pred, batch

    def build_dataset(self, dataset_path: Union[str, Path], mode: Mode):
        """Builds the dataset from the dataset path."""
        LOGGER.info(colorstr("bold", "red", f"Setting up {mode.name} dataset..."))

        if mode == Mode.TRAIN:
            if not self.trainset:
                dataset = self.get_dataset(
                    dataset_path, mode=Mode.TRAIN, use_cache=True, single_object=False
                )
                dataset = DatasetLM(bop_dataset=dataset, transforms=self.transforms)
                # dataset = DummyDataset(bop_dataset=dataset)
                self.trainset = dataset
            else:
                dataset = self.trainset
        elif mode == Mode.TEST:
            if not self.testset:
                dataset = self.get_dataset(
                    dataset_path, mode=Mode.TEST, use_cache=True, single_object=False
                )
                dataset = DatasetLM(bop_dataset=dataset, transforms=self.transforms)
                # dataset = DummyDataset(bop_dataset=dataset)
                self.testset = dataset
            else:
                dataset = self.testset
        return dataset

    def get_dataloader(self, dataset, batch_size=16, rank=0, mode=Mode.TRAIN):
        assert mode in [Mode.TRAIN, Mode.TEST]
        if dataset is None:
            with torch_distributed_zero_first(
                rank
            ):  # init dataset *.cache only once if DDP
                dataset = self.build_dataset(self.dataset_path, mode)
        shuffle = mode == Mode.TRAIN

        workers = self.args.workers if mode == Mode.TRAIN else self.args.workers * 2
        workers = 1 if self.args.debug else workers
        return build_dataloader(
            dataset, batch_size, workers, shuffle, rank
        )  # return dataloader

    def get_model(self, cfg=None, weights=None, verbose=True):
        # backbone = timm.create_model(
        #    self.args.backbone["type"],
        #    **self.args.backbone["init_cfg"]
        # )
        # backbone = hiera_base_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
        # freeze backbone
        backbone = timm.create_model(
            model_name="maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k",
            pretrained=True,
        )
        # if self.args.backbone["freeze"]:
        data_config = timm.data.resolve_model_data_config(backbone)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        if True:
            for param in backbone.parameters():
                param.requires_grad = False
        # model = GraphNet(
        #    backbone=backbone,
        #    in_channels=cfg.in_channels,
        #    out_channels=cfg.out_channels,
        #    channels=cfg.channels,
        #    n_res_blocks=cfg.n_res_blocks,
        #    attention_levels=cfg.attention_levels,
        #    attention_mode=AttentionMode.GAT if cfg.attention_mode == "gat" else None,
        #    channel_multipliers=cfg.channel_multipliers,
        #    unpooling_levels=cfg.unpooling_levels,
        #    n_heads=cfg.n_heads,
        #    d_cond=cfg.d_cond
        # )
        model = GraphNetv2(
            backbone=backbone,
            in_channels=3,
            out_channels=3,
            d_model=64,  # 128, #768,
            n_res_blocks=3,
            attention_levels=[1, 3, 4, 5],
            unpooling_levels=[2, 5, 6],
            channel_multipliers=[1, 2, 4, 4, 4, 8, 16, 32],
            n_heads=4,
            channels=16,
            d_cond=1024,
        )
        # print number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        print(
            f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        model = ModelWrapper(model)

        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        self.loss_names = ["loss"]
        return MposeValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args)
        )

    def preprocess_batch(self, batch):
        return {
            k: v.squeeze(0).to(self.device, non_blocking=True) for k, v in batch.items()
        }

    def criterion(self, pred, target):
        if self.i % 1000 == 0:
            np.save("temp/predx.npy", pred[0].detach().cpu().numpy())
            np.save("temp/targetx.npy", target["gt_features"].detach().cpu().numpy())
        self.i += 1
        pred_features = pred[0]#.unsqueeze(0)
        bs, num_vertices = pred_features.shape[:2]
        faces = pred[2]

        faces_single = faces[(faces[:, 0] < num_vertices) & (faces[:, 1] < num_vertices) & (faces[:, 2] < num_vertices)]
        faces_mul = faces_single.unsqueeze(0).repeat(bs, 1, 1)
        meshes = Meshes(verts=pred_features, faces=faces_mul)

        loss_chamfer, loss_normal = chamfer_distance(
            meshes.verts_padded(), target["gt_features"].squeeze(0),
            x_normals=meshes.verts_normals_padded(), y_normals=target["gt_normals"].squeeze(0),
        )
        

        # #            pcd.points = o3d.utility.Vector3dVector(vertices_combined)
        # #            pcd.normals = o3d.utility.Vector3dVector(normals_combined)
        # #            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        #edge_loss = torch.zeros(1).to(self.device)
        #laplace_loss = torch.zeros(1).to(self.device)
        edge_loss = mesh_edge_loss(meshes)
        laplace_loss = mesh_laplacian_smoothing(meshes)

        loss = (
            loss_chamfer + loss_normal + edge_loss + laplace_loss
        )  
        loss_list = torch.tensor(
            [
                loss_chamfer.sum().detach(),
                loss_normal.sum().detach(),
                laplace_loss.sum().detach(),
                edge_loss.sum().detach(),
            ]
        )
        return loss.sum() * self.args.batch, loss_list

    def plot_training_samples(self, batch, preds, ni):
        pass

    #        init_feats = batch[1].squeeze().cpu().numpy()
    #        init_edges = batch[2].squeeze().cpu().numpy()
    #        np.save("init.npy", init_feats)
    #        gt_feats = batch[3].squeeze().cpu().numpy()
    #        gt_normals = batch[4].squeeze().cpu().numpy()
    #        np.save("gt.npy", (gt_feats, gt_normals))
    #        pred_feats = preds[0].squeeze().detach().cpu().numpy()
    #        pred_edges = preds[1].squeeze().detach().cpu().numpy().T
    #        np.save("pred.npy", pred_feats)
    #        np.save("pred_edges.npy", pred_edges)
    #  gt_graph = Graph(feature_matrix=gt_feats, edge_index_list=gt_edges)
    #  gt_graph.visualize("gt_graph.png")
    #  pred_feats = preds[0].squeeze(0).detach().cpu().numpy()
    #  pred_edges = preds[1].squeeze(0).detach().cpu().numpy().T
    #  pred_graph = Graph(feature_matrix=pred_feats, edge_index_list=pred_edges)
    #  pred_graph.visualize("pred_graph.png")

    def progress_string(self) -> str:
        return ("\n" + "%11s" * (3 + 4)) % (  # len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            "lr",
            "chamfer",
            "normal",
            "edge",
            "laplace",
            #            "total",
            #            *self.loss_names,
        )
