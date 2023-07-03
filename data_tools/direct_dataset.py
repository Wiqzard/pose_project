
from typing import Optional, Union, List, Tuple, Callable
import struct

import cv2
from PIL import Image
import  numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import dill

from utils import LOGGER
from utils.bbox import Bbox
from utils.flags import Mode 
from data_tools.bop_dataset import BOPDataset


class DirectDataset(Dataset):
    def __init__(self, bop_dataset: BOPDataset, cfg=None, transforms=None, mode=None) -> None:
        super().__init__()
        #self.reduce = reduce
        self.reduce = 1
        #self.mae = mae
        self.dataset =  bop_dataset
        self.transforms = transforms
        self.args = cfg
        im_H,im_W = self.args.imgsz
        self.coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)
        self.model_points = None

        #self.augmentator = None
        #if self.cfg.DATASET.AUGMENT.AUG_TRAIN and flag == Flag.TRAIN and not mae:
        #    self._set_augmentator()
        
        
        
#    def _set_augmentator(self) -> Augmentator:
#        aug_cfg = self.cfg.DATASET.AUGMENT
#        augmentator = Augmentator(imgsz=self.cfg.DATASET.IMG_RES)
#        augmentator.add_box_augementator(
#            p=aug_cfg.BOX_AUG.P, dist_type=DistType.UNIFORM, sigma=aug_cfg.BOX_AUG.SIGMA
#        )
#        augmentator.add_color_augmentator(
#            p=aug_cfg.COLOR_AUG.P, aug_code=aug_cfg.COLOR_AUG.CODE
#        )
#        if bg_path := aug_cfg.BG_AUG.BG_PATH and aug_cfg.BG_AUG.P > 0:
#            augmentator.add_bg_augmentator(
#                p=aug_cfg.BG_AUG.P,
#                bg_path=bg_path,
#                im_H=self.cfg.DATASET.IMG_RES[0],
#                im_W=self.cfg.DATASET.IMG_RES[1],
#                truncate_fg=aug_cfg.BG_AUG.TRUNCATE_FG,
#            )
#        augmentator.add_mask_augmentator(p=aug_cfg.MASK_AUG.P)
#        augmentator.build()
#        self.augmentator = augmentator
#        return augmentator

    def __len__(self) -> int:
        return self.dataset.length() // self.reduce

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        idx *= self.reduce
        input_data = {
            "roi_cls": None,
            "roi_img": None,
            "bbox": None,
            "roi_coord_2d": None,
            "cams": None,
            #"roi_cams": None,
            "roi_centers": None,
            "roi_wh": None,
            "resize_ratios": None,
        }
        gt_data = {
            "gt_pose": None,
            "trans_ratio": None,
           # "roi_extents": None,
        }
        imgsz = self.args.imgsz
        input_res = self.args.inp_size
        

        img_path = self.dataset.get_img_path(idx)
        img = cv2.imread(str(img_path))

        if self.args.use_obj_bbox:
            bboxs = self.dataset.get_bbox_objs(idx)
        else:
            bboxs = self.dataset.get_bbox_visibs(idx)
        bbox = Bbox.from_xywh(bboxs[0])
#        if self.augmentator:
#            _, _, bboxs = self.augmentator(bbox=bboxs)
        roi_img = crop_square_resize(
                img,
                bbox,
                input_res,
            ).astype("uint8")

        if self.transforms is not None:
            roi_img = Image.fromarray(roi_img)  # cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            roi_img = self.transforms(roi_img)
        else:
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            roi_img = cv2.resize(roi_img, (224, 224))
            roi_img = torch.from_numpy(roi_img).float().permute(2, 0, 1)
            roi_img = roi_img / 255.0

      #  if self.augmentator:
      #      roi_img, mask_trunc, _ = self.augmentator(
      #          img=roi_img
      #      )
        # normalize after augmentation !!!

        #roi_img = standard_normalize(roi_img, use_0_255=True)
        roi_cls = self.dataset.get_obj_ids(idx)[0]  # List
        roi_coord_2d = self.get_roi_coord_2d(im_res=imgsz, out_res=input_res, bbox=bbox)

        cam = self.dataset.get_cam(idx).squeeze()

        #roi_cls = torch.tensor(roi_cls - 1, dtype=torch.int32)
        #roi_img = torch.from_numpy(roi_img).float()
        bboxs_tensor = torch.tensor([bbox.xyxy], dtype=torch.int32)
        roi_coord_2d = torch.from_numpy(roi_coord_2d).float()
        cam = torch.from_numpy(cam)
        roi_centers = torch.tensor(bbox.center, dtype=torch.float32)
        roi_wh = torch.tensor([bbox.w, bbox.h], dtype=torch.float32)

#        roi_extents = self.extents[0]  # [idx]
#        roi_extents = torch.tensor(roi_extents, dtype=torch.float32)
        resize_ratios = torch.tensor(
            self.args.inp_size / bbox.scale(*self.args.imgsz),
            dtype=torch.float32,
        )
        # deltas
        pose = self.dataset.get_poses(idx)[0]
        pose[:, 3] = pose[:, 3] / 1000    
        obj_center = self.get_2d_centroid(idx)
        delta_c = obj_center - bbox.center
        resize_ratio = self.args.inp_size / bbox.scale(
            *self.args.imgsz
        )
        z_ratio = pose[-1][-1] / resize_ratio
        trans_ratio = torch.tensor(
            [delta_c[0] / bbox.w, delta_c[1] / bbox.h, z_ratio],
            dtype=torch.float32,
        )
        roi_points = self.get_model_points(
            self.args.num_points
        )[roi_cls - 1]
        roi_points = torch.tensor(roi_points, dtype=torch.float32)
        if True: #self.mae:
            roi_img = torch.cat([roi_img, roi_coord_2d], dim=0)

        sym_info = [False]
        sym_info = torch.tensor(sym_info)
        input_data["roi_cls"] = roi_cls
        input_data["roi_img"] = roi_img
        input_data["bbox"] = bboxs_tensor
        input_data["roi_coord_2d"] = roi_coord_2d
        input_data["cams"] = cam
        input_data["roi_centers"] = roi_centers
        input_data["roi_wh"] = roi_wh
        input_data["resize_ratios"] = resize_ratios
        #input_data["roi_extents"] = roi_extents

        gt_data["gt_pose"] = torch.tensor(pose, dtype=torch.float32)
        gt_data["gt_points"] = roi_points
        gt_data["trans_ratio"] = trans_ratio
        #gt_data["roi_extents"] = roi_extents
        gt_data["symmetry_info"] = sym_info
        return input_data, gt_data


    def get_2d_centroid(self, idx: int) -> np.ndarray:
        """
        Get the 2D centroid of an object in an image. Project translation onto the image plane
        and normalize.

        Args:
            idx: The index of the image.

        Returns:
            An array containing the x and y coordinates of the 2D centroid.
        """
        poses = self.dataset.get_poses(idx)[0]
        cam = self.dataset.get_cam(idx)
        trans = poses[:3, 3]  / 1000  # convert to meters
        proj = (cam @ trans.T).T
        proj_2d = proj / proj[2:]
        return proj_2d[:-1]

    def get_roi_img(
        self,
        input_res: int,
        img: Optional[np.ndarray] = None,
        bbox_objs: Optional[Bbox] = None,
    ) -> list[np.ndarray]:
        """Get a list of region-of-interest images (ROI) from the input image. Two different
        cropping methods are supported: warp affine and crop and resize.
        Args:
            idx (Optional[int]): Index of the input image to extract ROIs from.
            normalize (Optional[bool]): Whether to normalize the ROIs using standard normalization.
            img (Optional[np.ndarray]): Input image from which to extract ROIs.
            bbox_objs (Optional[List[Bbox]]): List of bounding box objects corresponding to ROIs.

        Returns:
            List[np.ndarray]: List of ROIs as numpy arrays.

        Raises:
            ValueError: If both img and bbox_objs are None.

        """
        roi_imgs = [
            crop_square_resize(
                img,
                bbox_obj,
                input_res,
            )
            for bbox_obj in bbox_objs
        ]

        return roi_imgs

    def get_roi_coord_2d(self, im_res:Tuple[int, int],out_res:int, bbox: Bbox) -> np.ndarray:
        """Get 2D coordinates of region-of-interest (ROI) based on the input bounding box.

        Args:
            im_res (Tuple[int, int]): Resolution(Height, Width) of the input image.
            out_res (Tuple[int, int]): Resolution(Width) of the output image.
            bbox (Bbox): Bounding box object corresponding to the ROI.

        Returns:
            np.ndarray: 2D coordinates of the ROI as a numpy array.

        """
        im_H, im_W = im_res
        coord_2d = self.coord_2d
        if coord_2d is None:
            coord_2d = get_2d_coord_np(im_W, im_H, low=0, high=1).transpose(1, 2, 0)
        self.roi_coord_2d = crop_square_resize(
            coord_2d,
            bbox,
            crop_size=out_res,
        ).transpose(2, 0, 1)
        return self.roi_coord_2d

    def get_model_points(self, num_pm_points: int) -> np.ndarray:
        """
        Returns a dictionary containing a random subset of model points for each object in the YCB dataset.

        Args:
            num_pm_points (int): The maximum number of model points to keep for each object.

        Returns:
            np.ndarray: A dictionary where each key corresponds to an object in the dataset and the value is a NumPy array
            of shape (N_i, 3), where N_i is the number of model points kept for the object, and 3 represents the (x, y, z)
            coordinates of each point.
        """
        objs = self.dataset.model_names
        models = self.models

        cur_model_points = {}
        num = np.inf
        for i, obj_name in enumerate(objs):
            obj_name = 0
            model_points = models[obj_name]["pts"]
            cur_model_points[i] = model_points
            if model_points.shape[0] < num:
                num = model_points.shape[0]
        num = min(num, num_pm_points)
        for i in range(len(cur_model_points)):
            keep_idx = np.arange(num)
            np.random.shuffle(keep_idx)  # random sampling
            cur_model_points[i] = cur_model_points[i][keep_idx, :]
        return cur_model_points

    @property
    def models(self):
        """
        Load the models for the current dataset.

        Returns:
            A list of dictionaries, each containing information about a single model in the dataset.
        """
        cache_path = self.dataset.models_root / f"models_{self.dataset.dataset_name}.pkl"
        if self.model_points:
            return self.model_points

        if cache_path.exists() and self.dataset.use_cache:
            with open(cache_path, "rb") as f:
                self.model_points= dill.load(f)
            return self.model_points
        models = []
        for model_num in self.dataset.model_names:
            model = load_ply(
                self.dataset.models_root / f"obj_{model_num}.ply",
                vertex_scale=0.001, #self.scale_to_meter,
            )
            model["bbox3d_and_center"] = get_bbox3d_and_center(model["pts"])
            models.append(model)
        LOGGER.info(f"cache models to {cache_path}")
        with open(cache_path, "wb") as cache_path:
            dill.dump(models, cache_path)
        self.model_points = models
        return models


def crop_square_resize(
    img: np.ndarray, bbox: int, crop_size: int = None, interpolation=None
) -> np.ndarray:
    """
    Crop and resize an image to a square of size crop_size, centered on the given bounding box.

    Args:
    -----------
    img : numpy.ndarray
        Input image to be cropped and resized.
    bbox : int
        Bounding box coordinates of the object of interest. Must be in the format x1, y1, x2, y2.
    crop_size : int
        The size of the output square image. Default is None, which will use the largest dimension of the bbox as the crop_size.
    interpolation : int, optional
        The interpolation method to use when resizing the image. Default is None, which will use cv2.INTER_LINEAR.

    Returns:
    --------
    numpy.ndarray
        The cropped and resized square image.

    Raises:
    -------
    ValueError:
        If crop_size is not an integer.
    """

    if not isinstance(crop_size, int):
        raise ValueError("crop_size must be an int")

    x1, y1, x2, y2 = bbox.xyxy
    bw = bbox.w  # Bbox[2]
    bh = bbox.h  # Bbox[3]
    bbox_center = np.array(bbox.center)  # np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])

    if bh > bw:
        x1 = bbox_center[0] - bh / 2
        x2 = bbox_center[0] + bh / 2
    else:
        y1 = bbox_center[1] - bw / 2
        y2 = bbox_center[1] + bw / 2

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if img.ndim > 2:
        roi_img = np.zeros((max(bh, bw), max(bh, bw), img.shape[2]), dtype=img.dtype)
    else:
        roi_img = np.zeros((max(bh, bw), max(bh, bw)), dtype=img.dtype)
    roi_x1 = max((0 - x1), 0)
    x1 = max(x1, 0)
    roi_x2 = roi_x1 + min((img.shape[1] - x1), (x2 - x1))
    roi_y1 = max((0 - y1), 0)
    y1 = max(y1, 0)
    roi_y2 = roi_y1 + min((img.shape[0] - y1), (y2 - y1))
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    roi_img[roi_y1:roi_y2, roi_x1:roi_x2] = img[y1:y2, x1:x2].copy()
    roi_img = cv2.resize(roi_img, (crop_size, crop_size), interpolation=interpolation)
    return roi_img



def get_2d_coord_np(width, height, low=0, high=1, fmt="CHW", endpoint=False):
    """
    Args:
        width:
        height:
        endpoint: whether to include the endpoint
    Returns:
        xy: (2, height, width)
    """
    # coords values are in [low, high]  [0,1] or [-1,1]
    x = np.linspace(low, high, width, dtype=np.float32, endpoint=endpoint)
    y = np.linspace(low, high, height, dtype=np.float32, endpoint=endpoint)
    xy = np.asarray(np.meshgrid(x, y))
    if fmt == "HWC":
        xy = xy.transpose(1, 2, 0)
    elif fmt == "CHW":
        pass
    else:
        raise ValueError(f"Unknown format: {fmt}")
    return xy

def get_bbox3d_and_center(pts):
    """
    pts: Nx3
    ---
    bb: bb3d+center, 9x3
    """
    bb = []
    minx, maxx = min(pts[:, 0]), max(pts[:, 0])
    miny, maxy = min(pts[:, 1]), max(pts[:, 1])
    minz, maxz = min(pts[:, 2]), max(pts[:, 2])
    avgx = np.average(pts[:, 0])
    avgy = np.average(pts[:, 1])
    avgz = np.average(pts[:, 2])
    bb = np.array(
        [
            [maxx, maxy, maxz],
            [minx, maxy, maxz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, maxy, minz],
            [minx, maxy, minz],
            [minx, miny, minz],
            [maxx, miny, minz],
            [avgx, avgy, avgz],
        ],
        dtype=np.float32,
    )
    return bb


def load_ply(path, vertex_scale=1.0):
    f = open(path, "r")
    face_n_corners = 3
    n_pts = 0
    n_faces = 0
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False
    texture_file = None

    # Read the header.
    while True:
        # Strip the newline character(s)
        line = f.readline()
        if isinstance(line, str):
            line = line.rstrip("\n").rstrip("\r")
        else:
            line = str(line, "utf-8").rstrip("\n").rstrip("\r")

        if line.startswith("comment TextureFile"):
            texture_file = line.split()[-1]
        elif line.startswith("element vertex"):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith("element"):  # Some other element.
            header_vertex_section = False
            header_face_section = False
        elif line.startswith("property") and header_vertex_section:
            # (name of the property, data type)
            prop_name = line.split()[-1]
            if prop_name == "s":
                prop_name = "texture_u"
            if prop_name == "t":
                prop_name = "texture_v"
            prop_type = line.split()[-2]
            pt_props.append((prop_name, prop_type))
        elif line.startswith("property list") and header_face_section:
            elems = line.split()
            if elems[-1] == "vertex_indices" or elems[-1] == "vertex_index":
                # (name of the property, data type)
                face_props.append(("n_corners", elems[2]))
                for i in range(face_n_corners):
                    face_props.append(("ind_" + str(i), elems[3]))
            elif elems[-1] == "texcoord":
                # (name of the property, data type)
                face_props.append(("texcoord", elems[2]))
                for i in range(face_n_corners * 2):
                    face_props.append(("texcoord_ind_" + str(i), elems[3]))
            else:
                LOGGER.warning("Warning: Not supported face property: " + elems[-1])
        elif line.startswith("format"):
            if "binary" in line:
                is_binary = True
        elif line.startswith("end_header"):
            break

    # Prepare data structures.
    model = {}
    if texture_file is not None:
        model["texture_file"] = texture_file
    model["pts"] = np.zeros((n_pts, 3), np.float32)
    if n_faces > 0:
        model["faces"] = np.zeros((n_faces, face_n_corners), np.float32)

    pt_props_names = [p[0] for p in pt_props]
    face_props_names = [p[0] for p in face_props]

    is_normal = False
    if {"nx", "ny", "nz"}.issubset(set(pt_props_names)):
        is_normal = True
        model["normals"] = np.zeros((n_pts, 3), np.float32)

    is_color = False
    if {"red", "green", "blue"}.issubset(set(pt_props_names)):
        is_color = True
        model["colors"] = np.zeros((n_pts, 3), np.float32)

    is_texture_pt = False
    if {"texture_u", "texture_v"}.issubset(set(pt_props_names)):
        is_texture_pt = True
        model["texture_uv"] = np.zeros((n_pts, 2), np.float32)

    is_texture_face = False
    if {"texcoord"}.issubset(set(face_props_names)):
        is_texture_face = True
        model["texture_uv_face"] = np.zeros((n_faces, 6), np.float32)

    # Formats for the binary case.
    formats = {
        "float": ("f", 4),
        "double": ("d", 8),
        "int": ("i", 4),
        "uchar": ("B", 1),
    }

    # Load vertices.
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "red",
            "green",
            "blue",
            "texture_u",
            "texture_v",
        ]
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                read_data = f.read(format[1])
                val = struct.unpack(format[0], read_data)[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model["pts"][pt_id, 0] = float(prop_vals["x"])
        model["pts"][pt_id, 1] = float(prop_vals["y"])
        model["pts"][pt_id, 2] = float(prop_vals["z"])

        if is_normal:
            model["normals"][pt_id, 0] = float(prop_vals["nx"])
            model["normals"][pt_id, 1] = float(prop_vals["ny"])
            model["normals"][pt_id, 2] = float(prop_vals["nz"])

        if is_color:
            model["colors"][pt_id, 0] = float(prop_vals["red"])
            model["colors"][pt_id, 1] = float(prop_vals["green"])
            model["colors"][pt_id, 2] = float(prop_vals["blue"])

        if is_texture_pt:
            model["texture_uv"][pt_id, 0] = float(prop_vals["texture_u"])
            model["texture_uv"][pt_id, 1] = float(prop_vals["texture_v"])

    # Load faces.
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == "n_corners":
                    if val != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                        # print("Number of face corners: " + str(val))
                        # exit(-1)
                elif prop[0] == "texcoord":
                    if val != face_n_corners * 2:
                        raise ValueError("Wrong number of UV face coordinates.")
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == "n_corners":
                    if int(elems[prop_id]) != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                elif prop[0] == "texcoord":
                    if int(elems[prop_id]) != face_n_corners * 2:
                        raise ValueError("Wrong number of UV face coordinates.")
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model["faces"][face_id, 0] = int(prop_vals["ind_0"])
        model["faces"][face_id, 1] = int(prop_vals["ind_1"])
        model["faces"][face_id, 2] = int(prop_vals["ind_2"])

        if is_texture_face:
            for i in range(6):
                model["texture_uv_face"][face_id, i] = float(
                    prop_vals["texcoord_ind_{}".format(i)]
                )

    f.close()
    model["pts"] *= vertex_scale

    return model