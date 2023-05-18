from typing import Tuple, List, Dict, Union, Optional
from pathlib import Path

import numpy as np
import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data_tools.bop_dataset import BOPDataset, Flag, DatasetType
from data_tools.graph_tools.graph import Graph
from utils.bbox import Bbox


class LMDataset(Dataset):
    def __init__(self, bop_dataset: BOPDataset, cfg=None) -> None:
        super().__init__()
        self.dataset = bop_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        img = self.get_img(index)
        obj_ids = self.get_obj_ids(index)
        cam = self.get_cam(index)
        bbox_objs = self.get_bbox_objs(index)
        bbox_visibs = self.get_bbox_visibs(index)
        masks = self.get_masks(index)
        visib_masks = self.get_visib_masks(index)
        pose = self.get_poses(index)
        return self.dataset[index]

    def get_img(self, idx: int) -> np.ndarray:
        """
        Returns the image at a specified index in the dataset.

        Args:
            idx: The index of the image.

        Returns:
            The image as a numpy array of shape (H, W, C) in BGR channel order.

        Raises:
            ValueError: If the image has an unexpected number of channels.
        """
        path = Path(self.dataset.get_img_path(idx))
        # if not path.is_file():
        #    raise FileExistsError(f"image {path} does not exist")

        img = cv2.imread(str(path))
        if img.shape[-1] != 3:
            raise ValueError(f"image {path} has {img.shape[-1]} channels")
        return img

    def get_obj_ids(self, idx: int) -> List[int]:
        """
        Returns the object ids of the objects in the image at the given index.

        Args:
            idx (int): The index of the image.

        Returns:
            List[int]: A List of object ids.
        """
        return self.dataset.get_obj_ids(idx)

    def get_cam(self, idx: int) -> np.ndarray:
        """
        Get the camera matrix for the image at the given index.

        Args:
            idx (int): The index of the image.

        Returns:
            np.ndarray: The camera matrix.
        """
        return self.dataset.get_cam(idx)

    def get_bbox_objs(self, idx: int) -> List[Bbox]:
        """
        Get the bounding boxes for the objects in the image at the given index.

        Args:
            idx (int): The index of the image.

        Returns:
            List[Bbox]: A List of bounding boxes for each object in the image.
        """
        bbox_objs = self.dataset.get_bbox_objs(idx)
        bbox_objs = [Bbox.from_xywh(bbox_obj) for bbox_obj in bbox_objs]
        return bbox_objs

    def get_bbox_visibs(self, idx: int) -> List[Bbox]:
        """
        Get the visible bounding boxes for the objects in the image at the given index.

        Args:
            idx (int): The index of the image.

        Returns:
            List[Bbox]: A List of visible bounding boxes for each object in the image.
            H x W x 1
        """
        bbox_visibs = self.dataset.get_bbox_visibs(idx)
        bbox_visibs = [Bbox.from_xywh(bbox_visib) for bbox_visib in bbox_visibs]
        return bbox_visibs

    def get_masks(self, idx: int) -> List[np.ndarray]:
        """
        Get the masks for the objects in the image at the given index.

        Args:
            idx (int): The index of the image.

        Returns:
            List[np.ndarray]: A List of masks for each object in the image. H x W x 1
        """
        paths = list(self.dataset.get_mask_paths(idx))
        for path in paths:
            if not Path(path).is_file():
                raise FileExistsError(f"mask {path} does not exist")
        masks = [
            cv2.imread(str(path), 0)[..., None].astype("float32") / 255
            for path in paths
        ]

        return masks

    def get_visib_masks(self, idx: int) -> List[np.ndarray]:
        """
        Get the visible masks for the objects in the image at the given index.

        Args:
            idx (int): The index of the image.

        Returns:
            List[np.ndarray]: A List of visible masks for each object in the image.
        """
        paths = list(self.dataset.get_mask_visib_paths(idx))
        for path in paths:
            if not Path(path).is_file():
                raise FileExistsError(f"mask {path} does not exist")
        mask_visibs = [
            cv2.imread(str(path), 0)[..., None].astype("float32") / 255
            for path in paths
        ]
        return mask_visibs

    def get_poses(self, idx: int) -> np.ndarray:
        """
        Get the poses for the objects in the image at the given index.

        Args:
            idx (int): The index of the image.

        Returns:
            np.ndarray: A List of poses for each object in the image.
        """
        poses = self.dataset.get_poses(idx).copy()
        for pose in poses:
            pose[:3, 3] = pose[:3, 3] / 1000
        if len(poses) == 0:
            raise ValueError(f"no poses found for idx {idx}")
        return poses

    @property
    def extents(self) -> np.ndarray:
        """
        Return the extents (size) of each model in the dataset.

        Returns:
            An array containing the extents of each model, in the format [size_x, size_y, size_z].
        """
        return self.dataset.extents
