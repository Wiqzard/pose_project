
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import LOGGER
from engine.metrics import Metrics
from engine.validator import BaseValidator
from utils.pose_ops import pose_from_pred, pose_from_pred_centroid_z, get_rot_mat

from data_tools.direct_dataset import  DirectDataset
#from utils.annotator.annotator import Annotator

class HIERAValidator(BaseValidator):
    def __init__(
        self, dataloader=None, save_dir=None, pbar=None, cfg=None, _callbacks=None
    ):
        super().__init__(dataloader, save_dir, pbar, cfg, _callbacks)
        self.metrics = Metrics()

    def preprocess(self, batch):
        if isinstance(batch, list):

            batch_in, batch_gt = batch
           # coord2d = batch_in["roi_coord_2d"]
           # roi_img = batch_in["roi_img"]
           # batch_in["roi_img"] = torch.cat([roi_img, coord2d], dim=1)

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

    def postprocess(self, batch, input_data, gt_data=None):
        cams = input_data["cams"]
        roi_centers = input_data["roi_centers"]
        resize_ratios = input_data["resize_ratios"]
        roi_whs = input_data["roi_wh"]
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
                is_train=False,
            )
        elif self.args.trans_type == "trans":
            pred_ego_rot, pred_trans = pose_from_pred(
                pred_rot_m,
                pred_t_,
                eps=1e-4,
                is_allo="allo" in rot_type,
                is_train=False,
            )

        batch = {"rot": pred_ego_rot, "trans": pred_trans, "pred_t_": pred_t_}
        batch_c = batch.copy()
        input_data_c = input_data.copy()
        batch_c.update(input_data_c)
        batch_c = {k: v.cpu().detach() for k, v in batch_c.items()}

        return batch


    def init_metrics(self, model):
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.seen = 0
        self.jdict = []
        self.stats = []

    def get_desc(self):
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
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
        gt_data,
    ):
        # Metrics
        self.loss_dict = self.loss_dict if self.training else {"loss": torch.zeros((1))}
        self.metrics.update(self.loss_dict, gt_data, preds)
        bs = input_data["roi_cls"].shape[0]
        for si in range(bs):
            cls = input_data["roi_cls"][si].unsqueeze(0)
            rot = preds["rot"][si]
            trans = preds["trans"][si]
            # mask = batch["mask"][si]
            # full_mask = batch["full_mask"][si]
            # xyz, region
            self.seen += 1
            self.stats.append((cls, rot, trans))  # (conf, pcls, tcls)

            # Save
            # if self.cfg.SAVE_VAL:
            #    self.save_one(preds, input_data, gt_data, si)
            # if self.args.save_json:
            #    self.pred_to_json(predn, batch['im_file'][si])

            # if self.args.save_txt:
            #    file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
            #    self.save_one_txt(predn, self.args.save_conf, shape, file)

    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed

    def get_stats(self):
        # stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
        # if len(stats) and stats[0].any():
        #     self.metrics.process(*stats)
        self.nt_per_class = np.array(
            (1,)
        )  # np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        return self.metrics.results_dict

    def print_results(self):
        pf = (
            "%22s" + "%11i" * 2 + "%11.3g" * (len(self.metrics.keys) + 1)
        )  # print format
        LOGGER.info(
            pf
            % (
                "all",
                self.seen,
                self.nt_per_class.sum(),
                *self.metrics.avg_metrics,
                self.metrics.fitness,
            )
        )
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels"
            )
        # Print results per class
        # if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
        #    for i, c in enumerate(self.metrics.ap_class_index):
        #        LOGGER.info(
        #            pf
        #            % (
        #                self.names[c],
        #                self.seen,
        #                self.nt_per_class[c],
        #                *self.metrics.class_result(i),
        #            )
        #        )

    def save_evaluation_csv(self, preds, input_data, gt_data, si) -> None:
        """Save evaluation results to a csv file."""
        if self.cfg.SAVE_VAL:
            save_dir = Path(self.cfg.SAVE_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            csv_path = save_dir / "eval_results.csv"

            # self.metrics.save_results_csv(csv_path)

    def get_dataloader(self, flag):
        return_oimg = self.cfg.SAVE_VIDEO and not self.training
        dataset = BMW_Dataset(
            cfg=self.cfg, path=self.cfg.DATASET_ROOT, flag=flag, reduce=1, return_oimg=return_oimg
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.SOLVER.IMS_PER_BATCH,
            shuffle=False,
            num_workers=16,
        )
        return dataloader

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