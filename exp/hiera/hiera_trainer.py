import os
from typing import Any, Optional, Union
from pathlib import Path

import torch
import timm

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils import LOGGER, RANK, colorstr
from utils.torch_utils import (
    torch_distributed_zero_first,
    build_dataloader)
    
from utils.flags import Mode
from utils.pose_ops import pose_from_pred, pose_from_pred_centroid_z, get_rot_mat


from .hiera_validator import HIERAValidator
from engine.trainer import BaseTrainer
#from engine.optimizer.ranger import Ranger
#from utils.torch_utils import attempt_load_one_weight
from engine.losses.losses import (
    get_losses_names,
    compute_rot_loss,
    compute_point_matching_loss,
    compute_centroid_loss,
    compute_z_loss,
    compute_trans_loss,
)
from data_tools.bop_dataset import BOPDataset
from data_tools.direct_dataset import DirectDataset
#from models.hiera.hiera import HieraRaw
from models.heads.direct import PoseRegression

class HIERATrainer(BaseTrainer):
    def __init__(self, args):
        super(HIERATrainer, self).__init__(args)
        self.loss_names = get_losses_names(args)
        if RANK in (-1, 0):
            LOGGER.info(f"Selected Losses: {self.loss_names}") 
        self.train_dataset, self.eval_dataset, self.test_dataset = None, None, None


    def preprocess_batch(self, batch: Union[tuple[dict], list[dict], dict]) -> Any:
        """Preprocess batch for training/evaluation. Move to device."""
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch_in, batch_gt = batch
            batch_in = {
                k: v.to(self.device, non_blocking=True) for k, v in batch_in.items()
            }
            batch_gt = {
                k: v.to(self.device, non_blocking=True) for k, v in batch_gt.items()
            }
            batch = (batch_in, batch_gt)
        elif isinstance(batch, dict):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        else:
            raise ValueError("Batch type not supported")
        return batch

    def postprocess_batch(
        self,
        batch: Union[tuple[dict], list[dict], dict],
        input_data: dict[torch.Tensor],
        train: bool = False,
    ) -> Any:
        cams = input_data[0]["cams"]

        roi_centers = input_data[0]["roi_centers"]
        resize_ratios = input_data[0]["resize_ratios"]
        roi_whs = input_data[0]["roi_wh"]
        pred_rot_ = batch[0]
        pred_t_ = batch[1]
        rot_type = self.args.rot_type
        pred_rot_m = get_rot_mat(pred_rot_, rot_type)
        if self.args.trans_type == "centroid_z":
            pred_ego_rot, pred_trans = pose_from_pred_centroid_z(
                pred_rot_m,
                pred_centroids=pred_t_[:, :2],
                pred_z_vals=pred_t_[:, 2:3],  # must be [B, 1]
                roi_cams=cams,  # roi_cams,
                roi_centers=roi_centers,
                resize_ratios=resize_ratios,
                roi_whs=roi_whs,
                eps=1e-4,
                is_allo="allo" in rot_type,
                z_type=self.args.z_type,
                is_train=True,
            )
        elif self.args.trans_type == "trans":
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m,
                pred_t_,
                eps=1e-4,
                is_allo="allo" in rot_type,
                is_train=True,
            )

        preds = {"rot": pred_ego_rot, "trans": pred_trans, "pred_t_": pred_t_}
        batch = {k: v for k, v in input_data[0].items() if k != "roi_img"}
        batch["sym_infos"] =  [self.trainset.sym_infos[obj_id_.item()].to(self.device) for obj_id_ in batch["roi_cls"]]
        batch.update(input_data[1])
        return preds, batch

    def get_dataset(
        self, img_path, mode=Mode.TRAIN, use_cache=True, single_object=True
    ):
        return BOPDataset(img_path, mode, use_cache=True, single_object=True)

    def build_dataset(self, dataset_path: Union[str, Path], mode: Mode):
        """Builds the dataset from the dataset path."""
        LOGGER.info(colorstr("bold", "red", f"Setting up {mode.name} dataset..."))

        if mode == Mode.TRAIN:
            if not self.trainset:
                dataset = self.get_dataset(
                    dataset_path, mode=Mode.TRAIN, use_cache=True, single_object=True
                )
                dataset = DirectDataset(bop_dataset=dataset, cfg=self.args, transforms=self.transforms, reduce=self.args.reduce)
                self.trainset = dataset
            else:
                dataset = self.trainset
        elif mode == Mode.TEST:
            if not self.testset:
                dataset = self.get_dataset(
                    dataset_path, mode=Mode.TEST, use_cache=True, single_object=True
                )
                dataset = DirectDataset(bop_dataset=dataset, cfg=self.args, transforms=self.transforms, reduce=self.args.reduce)
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
  

    def get_model(self, weights, verbose:bool):
        """Build model and load pretrained weights if specified."""
        if isinstance(
            self.model, torch.nn.Module
        ):  # if model is loaded beforehand. No setup needed
            return
        ckpt = None
        if False: #str(self.model).endswith(".pt"):
            #model, ckpt = attempt_load_one_weight(self.model)
            cfg = ckpt["cfg"] if "cfg" in ckpt else None
        else:
            #model = HieraRaw(in_chans=5)  # build_model(self.args)
            backbone = timm.create_model(
                'mvitv2_base.fb_in1k',
                pretrained=True,
                num_classes=0,  # remove classifier nn.Linear
                in_chans=5
            )
            data_config = timm.data.resolve_model_data_config(backbone)
            self.transforms = timm.data.create_transform(**data_config, is_training=False)
            model = PoseRegression(backbone=backbone, emb_dim=768)
        self.model = model
        return model #ckpt

    def get_validator(self) -> Any:
        return HIERAValidator(
            dataloader=self.test_loader, save_dir=self.save_dir, cfg=self.args
        )

    def criterion(self, pred: dict, gt: dict) -> dict:
        out_rot = pred["rot"]
        out_trans = pred["trans"]
        out_centroid = pred["pred_t_"][:, :2]
        out_trans_z = pred["pred_t_"][:, 2]

        obj_id = gt["roi_cls"]
        gt_trans = gt["gt_pose"][:, :3, 3]
        gt_rot = gt["gt_pose"][:, :3, :3]
        gt_trans_ratio = gt["trans_ratio"]
        gt_points = gt["gt_points"]
        sym_infos = gt["sym_infos"]
        #extents = gt["roi_extents"]
        loss_dict = {}
        if self.args.pm_lw > 0:
            loss_pm = compute_point_matching_loss(
                args=self.args,
                out_rot=out_rot,
                gt_rot=gt_rot,
                gt_points=gt_points,
                out_trans=out_trans,
                gt_trans=gt_trans,
                extents=None, #extents,
                sym_infos=sym_infos,
            )
            loss_dict.update({**loss_pm})

        if self.args.rot_lw > 0:
            loss_rot = compute_rot_loss(
                args=self.args, out_rot=out_rot, gt_rot=gt_rot
            )
            loss_dict.update({"loss_rot": loss_rot})

        if self.args.centroid_lw > 0 and self.args.trans_type == "centroid_z":
            loss_centroid = compute_centroid_loss(
                args=self.args,
                out_centroid=out_centroid,
                gt_trans_ratio=gt_trans_ratio,
            )
            loss_dict.update({"loss_centroid": loss_centroid})

        if self.args.z_lw > 0:
            z_type = self.args.z_type
            if z_type == "rel":
                gt_z = gt_trans_ratio[:, 2]
            elif z_type == "abs":
                gt_z = gt_trans[:, 2]
            loss_z = compute_z_loss(
                args=self.args, out_trans_z=out_trans_z, gt_z=gt_z
            )
            loss_dict.update({"loss_z": loss_z})
        return sum(loss_dict.values()), torch.tensor(list(loss_dict.values()))

    def progress_string(self) -> str:
        return ("\n" + "%11s" * (3 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            "lr",
            *self.loss_names,
        )
