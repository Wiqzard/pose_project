
from pathlib import Path

from utils import SimpleClass


class PoseMetrics(SimpleClass):
    
    def __init__(self, save_dir=Path('.'), plot=False, on_plot=None, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

    def process(self, tp, conf, pred_cls, target_cls):
        """Process predicted results for object detection and update metrics."""
        results = ap_per_class(tp,
                               conf,
                               pred_cls,
                               target_cls,
                               plot=self.plot,
                               save_dir=self.save_dir,
                               names=self.names,
                               on_plot=self.on_plot)[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ['fitness'], self.mean_results() + [self.fitness]))

        
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import torch
from torch import nn
import numpy as np

from engine.losses.rot_loss import angular_distance


@dataclass
class Metrics:
    """Pose metrics class to store metrics for each eval."""

    losses: list[float] = field(default_factory=list)
    ang_distances: list[float] = field(default_factory=list)
    trans_distances: list[float] = field(default_factory=list)

    def update(
        self,
        loss_dict: dict[str, torch.Tensor],
        gt_data: dict[str, torch.Tensor],
        preds: dict[str, torch.Tensor],
    ) -> None:
        """Update the metrics with new data."""

        # loss_items = {k: v.clone().detach() for k, v in loss_dict.items()}
        loss = sum(loss_dict.values())

        pred_rot = preds["rot"]
        pred_trans = preds["trans"]  # pred_t_ (bs, 3)

        gt_rot = gt_data["gt_pose"][:, :3, :3]
        gt_trans = gt_data["gt_pose"][:, :3, 3]

        ang_distance = angular_distance(pred_rot, gt_rot)  # mean of batch
        eucl_distance = torch.norm(
            (pred_trans - gt_trans), p=2, dim=1
        ).mean()  # mean of batch

        self.losses.append(loss.item())
        self.ang_distances.append(ang_distance.item())
        self.trans_distances.append(eucl_distance.item())

    def reset(self) -> None:
        self.losses = []
        self.ang_distances = []
        self.trans_distances = []

    @property
    def avg_losses(self) -> dict[str, float]:
        return {
            "avg_loss": self.avg_loss,
            "ang_distance": self.avg_ang_distance,
            "trans_distance": self.avg_trans_distance,
        }

    @property
    def avg_loss(self) -> float:
        return sum(self.losses) / len(self.losses)

    @property
    def avg_ang_distance(self) -> float:
        return sum(self.ang_distances) / len(self.ang_distances)

    @property
    def avg_trans_distance(self) -> float:
        return sum(self.trans_distances) / len(self.trans_distances)

    @property
    def keys(self) -> list[str]:
        return ["total_loss", "angular_distance", "translation_distance"]

    @property
    def avg_metrics(self) -> list[float]:
        """Return the metrics."""
        return [self.avg_loss, self.avg_ang_distance, self.avg_trans_distance]

    @property
    def empty_dict(self) -> list[float]:
        """Return empty metrics."""
        return dict(zip(self.keys, [0.0, 0.0, 0.0]))

    @property
    def results_dict(self) -> dict[str, float]:
        """Return the metrics as a dictionary."""
        results = dict(zip(self.keys, self.avg_metrics))
        results.update({"fitness": self.fitness})
        return results

    @property
    def fitness(self) -> float:
        """Return the fitness of the metrics."""
        w = np.array([0.00, 0.1, 10])  # weights for angular, translation, loss
        return (1 / np.array(self.avg_metrics).copy() * w).sum() / 100

    def __str__(self) -> str:
        """Return the string representation of the metrics."""
        return f" Total Loss: {self.avg_loss:.4f},\n Angular Distance: {self.avg_ang_distance:.4f},\n Translation (L2): {self.avg_trans_distance:.4f}"

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}"
        )
