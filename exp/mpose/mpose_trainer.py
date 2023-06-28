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
    ModelWrapper
)
from engine.trainer import BaseTrainer
from utils import colorstr
from data_tools.graph_tools.graph import Graph
from data_tools.bop_dataset import BOPDataset
from data_tools.dummy_dataset import DummyDataset
from data_tools.dataset import DatasetLM
from utils.flags import Mode
#import hiera
from models.backbones.hiera import hiera_base_224

from gat_inf import GraphNet, AttentionMode

from exp.mpose.mpose_validator import MposeValidator

from pytorch3d.loss import chamfer_distance
import torch
from torch.nn import functional as F


def compute_edge_lengths(feature_matrix, edge_index):
    """
    Computes the lengths of edges in a graph
    Args:
    feature_matrix (torch.Tensor): Tensor of shape (N, 3) representing the 3D coordinates of nodes in the graph
    edge_index (torch.Tensor): Tensor of shape (E, 2) representing pairs of nodes that form edges
    Returns:
    torch.Tensor: Tensor of shape (E,) representing the lengths of edges in the graph
    """
    source_nodes = feature_matrix[edge_index[0,:]]
    target_nodes = feature_matrix[edge_index[1,:]]
    edge_lengths = torch.norm(source_nodes - target_nodes, dim=1)
    return edge_lengths

def edge_loss(feature_matrix1, edge_index1, feature_matrix2, edge_index2):
    """
    Computes the MSE loss of edge lengths between two graphs
    Args:
    feature_matrix1, feature_matrix2 (torch.Tensor): Tensors of shape (N, 3) representing the 3D coordinates of nodes in the graphs
    edge_index1, edge_index2 (torch.Tensor): Tensors of shape (E, 2) representing pairs of nodes that form edges in the graphs
    Returns:
    torch.Tensor: Scalar tensor representing the loss
    """
    edge_lengths1 = compute_edge_lengths(feature_matrix1, edge_index1)
    edge_lengths2 = compute_edge_lengths(feature_matrix2, edge_index2)

    # If the number of edges in the two graphs are different, we need to handle that.
    # Here we just pad the shorter edge_lengths tensor with zeros.
    if edge_lengths1.shape != edge_lengths2.shape:
     raise ValueError("The two graphs have different number of edges")
     #   max_len = max(edge_lengths1.shape[0], edge_lengths2.shape[0])
     #   if edge_lengths1.shape[0] < max_len:
     #       edge_lengths1 = F.pad(edge_lengths1, (0, max_len - edge_lengths1.shape[0]))
     #   else:
     #       edge_lengths2 = F.pad(edge_lengths2, (0, max_len - edge_lengths2.shape[0]))

    loss = F.mse_loss(edge_lengths1, edge_lengths2)
    return loss


class MposeTrainer(BaseTrainer):
    def get_dataset(self, img_path, mode=Mode.TRAIN, use_cache=True, single_object=False):
        return BOPDataset(img_path, mode, use_cache=True, single_object=False)
        

    def build_dataset(self, dataset_path:Union[str, Path], mode:Mode):
        """Builds the dataset from the dataset path."""
        LOGGER.info(colorstr("bold", "red", f"Setting up {mode.name} dataset..."))
        
        if mode == Mode.TRAIN:
            if not self.trainset:
                dataset = self.get_dataset(dataset_path, mode=Mode.TRAIN, use_cache=True, single_object=False)
                dataset = DatasetLM(bop_dataset=dataset)
                #dataset = DummyDataset(bop_dataset=dataset)
                self.trainset = dataset
            else:
                dataset = self.trainset
        elif mode == Mode.TEST:
            if not self.testset:
                dataset = self.get_dataset(dataset_path, mode=Mode.TEST, use_cache=True, single_object=False)
                dataset = DatasetLM(bop_dataset=dataset)
                #dataset = DummyDataset(bop_dataset=dataset)
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

        #backbone = timm.create_model(
        #    self.args.backbone["type"],
        #    **self.args.backbone["init_cfg"]
        #)
        backbone = hiera_base_224(pretrained=True, checkpoint="mae_in1k_ft_in1k")
        # freeze backbone
        if self.args.backbone["freeze"]:
            for param in backbone.parameters():
                param.requires_grad = False 
            
        model = GraphNet(
            backbone=backbone,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            channels=cfg.channels,
            n_res_blocks=cfg.n_res_blocks,
            attention_levels=cfg.attention_levels,
            attention_mode=AttentionMode.GAT if cfg.attention_mode == "gat" else None,
            channel_multipliers=cfg.channel_multipliers,
            unpooling_levels=cfg.unpooling_levels,
            n_heads=cfg.n_heads,
            d_cond=cfg.d_cond
        )
        # print number of parameters
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        model = ModelWrapper(model)
        
        if weights:
            model.load(weights)
        return model
    

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        self.loss_names = ["loss"]
        return MposeValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def preprocess_batch(self, batch):
        batch[1] = batch[1].squeeze(0)
        batch[2] = batch[2].squeeze(0)
        batch[3] = batch[3].squeeze(0)
        batch[4] = batch[4].squeeze(0)
        return [item.to(self.device, non_blocking=True) for item in batch]

    def criterion(self, pred, target):
        #com = (pred[0].mean(dim=0) - target[3].mean(dim=0))**2
        #MSE
        #edge_length_loss = edge_loss(feature_matrix1=pred[0], edge_index1=pred[1], feature_matrix2=target[3], edge_index2=target[4])
       # split_pred = pred[0].view(pred[0].shape[0]//64, -1, 3)
       # split_target = target[3].view(target[3].shape[0]//64, -1, 3)
        #com_loss = torch.mean((split_pred.mean(dim=1) - split_target.mean(dim=1))**2, dim=1)/split_pred.shape[0] # / 64 
      #  if com_loss.sum() > 1000:
      #      print(split_pred.mean(dim=1))
      #      print(split_target.mean(dim=1))
       #     com_loss = torch.zeros_like(com_loss).to(com_loss.device)# tensor([0])
        pred_features = pred[0].unsqueeze(0)

#            pcd.points = o3d.utility.Vector3dVector(vertices_combined)
#            pcd.normals = o3d.utility.Vector3dVector(normals_combined)
#            o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        target_features = target[3].unsqueeze(0)
        loss_chamfer = chamfer_distance(pred_features, target_features)[0]# / 100000
        #loss_chamfer = chamfer_distance(split_pred, split_target)[0]# / 100000
        loss =  loss_chamfer #com_loss#+ edge_length_loss
        #loss_chamfer = chamfer_distance(pred[0].unsqueeze(0), target[3].unsqueeze(0))[0]
        #loss = loss_chamfer #com #torch.mean((pred[0] - target[3]) ** 2) 
        return loss.sum() * self.args.batch, torch.tensor([ loss_chamfer.sum().detach()])#, edge_length_loss.sum().detach()])com_loss.sum().detach() ,

        
    def plot_training_samples(self, batch, preds, ni):
        init_feats = batch[1].squeeze().cpu().numpy()
        init_edges = batch[2].squeeze().cpu().numpy()
        np.save("init.npy", init_feats)
        gt_feats = batch[3].squeeze().cpu().numpy()
        gt_normals = batch[4].squeeze().cpu().numpy()
        np.save("gt.npy", (gt_feats, gt_normals))
        pred_feats = preds[0].squeeze().detach().cpu().numpy()
        pred_edges = preds[1].squeeze().detach().cpu().numpy().T
        np.save("pred.npy", pred_feats)
        np.save("pred_edges.npy", pred_edges)
      #  gt_graph = Graph(feature_matrix=gt_feats, edge_index_list=gt_edges)
      #  gt_graph.visualize("gt_graph.png")
      #  pred_feats = preds[0].squeeze(0).detach().cpu().numpy()
      #  pred_edges = preds[1].squeeze(0).detach().cpu().numpy().T
      #  pred_graph = Graph(feature_matrix=pred_feats, edge_index_list=pred_edges)
      #  pred_graph.visualize("pred_graph.png")
        
    
    def progress_string(self) -> str:
        return ("\n" + "%11s" * (5 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            "lr",
            "com",
            "chamfer",
            "edge"
#            "total",
#            *self.loss_names,
        )