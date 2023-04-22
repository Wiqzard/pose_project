from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LinemodDataset(Dataset):
    def keypoints_to_map(self, mask, pts2d, unit_vectors=True):
        # based on: https://github.com/zju3dv/pvnet/blob/master/lib/datasets/linemod_dataset.py
        mask = mask[0]
        h, w = mask.shape
        n_pts = pts2d.shape[0]
        xy = np.argwhere(mask == 1.)[:, [1, 0]]
        xy = np.expand_dims(xy.transpose(0, 1), axis=1)
        pts_map = np.tile(xy, (1, n_pts, 1))
        pts_map = np.tile(np.expand_dims(pts2d, axis=0), (pts_map.shape[0], 1, 1)) - pts_map
        if unit_vectors:
            norm = np.linalg.norm(pts_map, axis=2, keepdims=True)
            norm[norm < 1e-3] += 1e-3
            pts_map = pts_map / norm
        pts_map_out = np.zeros((h, w, n_pts, 2), np.float32)
        pts_map_out[xy[:, 0, 1], xy[:, 0, 0]] = pts_map
        pts_map_out = np.reshape(pts_map_out, (h, w, n_pts * 2))
        pts_map_out = np.transpose(pts_map_out, (2, 0, 1))
        return pts_map_out

        
    def __getitem__(self, idx:int) -> Any:
        # return roi image, roi mask, keypoints, bbox, R, t
        raise NotImplementedError
        

class DummyDataset:
    def __init__(self) -> None:
        self.len = 10
        self.imgsz = 640
        self.num_keypoints = 10
        self.num_classes = 5
        self.torch = True 

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx:int) -> Any:
        # return roi image, roi mask, keypoints, bbox, R, t
        img = np.random.rand(self.imgsz, self.imgsz, 3)
        #img = torch.randn(3, self.imgsz, self.imgsz)
        mask = np.random.rand(self.imgsz, self.imgsz)
        mask = (mask > 0.5).astype(np.uint8)
        keypoints = np.random.rand(10, 2)
        bbox = np.random.rand(4)
        class_id = np.random.randint(0, self.num_classes)
        R = np.random.rand(3, 3)
        t = np.random.rand(3)
        if torch:
            img = torch.from_numpy(img).float()
            mask = torch.from_numpy(mask).float()
            keypoints = torch.from_numpy(keypoints).float()
            bbox = torch.from_numpy(bbox).float()
            class_id = torch.from_numpy(class_id).float()
            R = torch.from_numpy(R).float()
            t = torch.from_numpy(t).float()
        return img, mask, keypoints, bbox, class_id,  R, t 

dummy_dataset = DummyDataset()
dummy_dataloader = DataLoader(dummy_dataset, batch_size=2, shuffle=True, num_workers=0) 

# whole image. set detach graph of weights where attention is below threshold
