from typing import Any, Union
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import LOGGER
from utils.flags import Mode
from engine.metrics import Metrics
from engine.validator import BaseValidator
from utils.pose_ops import pose_from_pred, pose_from_pred_centroid_z, get_rot_mat
from engine.losses.losses import (
    get_losses_names,
    compute_rot_loss,
    compute_point_matching_loss,
    compute_centroid_loss,
    compute_z_loss,
    compute_trans_loss,
)
from data_tools.direct_dataset import DirectDataset
from data_tools.bop_dataset import BOPDataset

# from utils.annotator.annotator import Annotator


class HIERAValidator(BaseValidator):
    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, cfg=None, _callbacks=None
    ):
        super().__init__(dataloader, save_dir, pbar, cfg, _callbacks)
        self.metrics = Metrics(args=self.args)

    def preprocess(self, batch: Union[tuple[dict], list[dict], dict]) -> Any:
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

    def postprocess(
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
        # add .cpu() to avoid memory leak
        preds = {k: v.detach() for k, v in preds.items()}
        batch = {
            k: v.detach()
            for k, v in [*input_data[0].items(), *input_data[1].items()]
            if k != "roi_img"
        }
        batch["sym_infos"] = [
            self.testset.sym_infos[obj_id_.item()].to(self.device)
            for obj_id_ in batch["roi_cls"]
        ]
        return preds, batch

    def init_metrics(self):
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Targets",
            "Pose(TLoss",
            "Ang",
            "Trans",
            "Fitness)",
            # ADD-A, S
        )

    def update_metrics(
        self,
        preds,
        input_data,
    ):
        loss = self.loss if self.training else {"loss": torch.zeros((1))}
        self.metrics.update(loss, preds, input_data)
        bs = input_data["roi_cls"].shape[0]
        self.seen += bs

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed

    def print_results(self):
        pf = (
            "%22s" + "%11i" * 2 + "%11.3g" * (len(self.metrics.keys) + 1)
        )  # print format
        LOGGER.info(
            pf
            % (
                "all",
                self.metrics.num_targets,
                self.seen,
                *self.metrics.avg_metrics,
                self.metrics.fitness,
            )
        )
        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for obj_id, metric_list in self.metrics.avg_metrics_cls.items():
                LOGGER.info(
                    pf
                    % (
                        obj_id,
                        self.seen,
                        self.metrics.num_targets_cls[obj_id],
                        *self.metric_list,
                    )
                )

            # self.metrics.save_results_csv(csv_path)

    def criterion(self, pred: dict, gt: dict) -> dict:
        out_rot = pred["rot"]
        out_trans = pred["trans"]
        out_centroid = pred["pred_t_"][:, :2]
        out_trans_z = pred["pred_t_"][:, 2]

        gt_trans = gt["gt_pose"][:, :3, 3]
        gt_rot = gt["gt_pose"][:, :3, :3]
        gt_trans_ratio = gt["trans_ratio"]
        gt_points = gt["gt_points"]
        sym_infos = gt["sym_infos"]
        # extents = gt["roi_extents"]
        loss_dict = {}
        if self.args.pm_lw > 0:
            loss_pm = compute_point_matching_loss(
                args=self.args,
                out_rot=out_rot,
                gt_rot=gt_rot,
                gt_points=gt_points,
                out_trans=out_trans,
                gt_trans=gt_trans,
                extents=None,  # extents,
                sym_infos=sym_infos,
            )
            loss_dict.update({**loss_pm})

        if self.args.rot_lw > 0:
            loss_rot = compute_rot_loss(args=self.args, out_rot=out_rot, gt_rot=gt_rot)
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
            loss_z = compute_z_loss(args=self.args, out_trans_z=out_trans_z, gt_z=gt_z)
            loss_dict.update({"loss_z": loss_z})
        return sum(loss_dict.values()), torch.tensor(
            list(loss_dict.values()), device=self.device, requires_grad=False
        )

    def get_dataset(self, img_path, mode=Mode.TEST, use_cache=True, single_object=True):
        return BOPDataset(img_path, mode, use_cache=True, single_object=True)

    def get_dataloader(self, mode):
        # return_oimg = self.cfg.SAVE_VIDEO and not self.training
        dataset = self.get_dataset(
            dataset_path=self.args.dataset_path, use_cache=True, mode=mode
        )
        dataset = DirectDataset(bop_dataset=dataset, cfg=self.args, transforms=None)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.bs,
            shuffle=False,
            num_workers=16,
        )
        return dataloader

    def save_evaluation_csv(self, preds, input_data, gt_data, si) -> None:
        """Save evaluation results to a csv file."""
        if self.cfg.SAVE_VAL:
            save_dir = Path(self.cfg.SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            csv_path = save_dir / "eval_results.csv"

    def save_csv(self, preds, batch, dt):
        bs = batch["roi_cls"].shape[0]
        time = dt[0] + dt[1] + dt[2] / bs
        for i in range(bs):
            R = preds["rot"][i].cpu().numpy()
            trans = preds["trans"].cpu().numpy()

    def write_bop_results(
        self,
        scene_id: int,
        img_id: int,
        obj_id: int,
        score: float,
        time: float,
        R: list[float],
        t: list[float],
    ) -> None:
        """Write BOP results to a file."""
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            header = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
            with open(self.csv_path, mode="w+", newline="") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                writer.writerow(header)
        with open(self.csv_path, mode="a", newline="") as csv_file:
            writer = csv.writer(
                csv_file,
                delimiter=",",
            )
            R = " ".join(str(item) for item in R)
            t = " ".join(str(item) for item in t)
            writer.writerow(
                [int(scene_id), int(img_id), int(obj_id), score, R, t, time]
            )

    def setup_annotator(
        self, width, height, vis_masks=False, vis_poses=False, models_path=None
    ):
        # vis_orig_color=False for estimated poses?
        self.annotator = Annotator(
            width,
            height,
            vis_masks=vis_masks,
            vis_poses=vis_poses,
            models_path=models_path,
            vis_orig_color=False,
        )  # vis_poses=self.cfg.VIS_POSES)
