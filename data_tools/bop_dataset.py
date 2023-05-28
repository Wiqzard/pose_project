from typing import Any, Optional, Tuple, Union, List, Dict
from pathlib import Path
import enum
import json
import pickle
import json
from tqdm import tqdm
import copy
import struct
import open3d as o3d

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_tools.graph_tools.graph import Graph

from utils import LOGGER, RANK



class Flag(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()


class DatasetType(enum.Enum):
    BOP = enum.auto()
    LINEMOD = enum.auto()
    CUSTOM = enum.auto()


def require_dataset(method):
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_dataset"):
            return method(self, *args, **kwargs)
        else:
            raise AttributeError(
                "Dataset not loaded yet. Call _create_dataset() first."
            )

    return wrapper


def correct_suffix(path: Path) -> Path:
    """
    Correct the suffix of a path to be .png or .jpg.
    Args:
        path: The path to correct.
    Returns:
        The corrected path.
    """
    path = Path(path)
    if path.is_file():
        return path
    path = path.with_suffix(".png")
    if path.is_file():
        return path
    path = path.with_suffix(".jpg")
    if not path.is_file():
        raise FileNotFoundError(f"Mask {path} does not exist.")
    return path


class BOPDataset:
    """
    _dataset: dict of {}

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        path: str,
        flag: Flag,
        dataset_type: DatasetType = DatasetType.LINEMOD,
        use_cache: bool = False,
        single_object: Optional[bool] = False,
    ) -> None:
        self._path = Path(path)
        self.use_cache = use_cache
        self.flag = flag
        self.dataset_type = dataset_type
        self.dataset_name = self.dataset_type.name.lower()
        self.scale_to_meter = 0.001
        self.single_object = single_object

        if not self._path.is_dir():
            raise FileNotFoundError(f"Path {self._path} does not exist.")

        if self.flag == Flag.TRAIN:
            self.split_path = self._path / "train_pbr"
            self.models_root = self._path / "models"
        elif self.flag == Flag.VAL:
            raise ValueError("Validation set not implemented yet.")
        elif self.flag == Flag.TEST:
            self.split_path = self._path / "test"
            self.models_root = self._path / "models_eval"
        # self.split_graphs_path = self.split_path / "graphs"

        self.model_names = [
            f"{int(file.stem.split('_')[-1]):06d}"
            for file in self.models_root.glob("obj_*.ply")
        ]
        self.scene_paths = [
            folder
            for folder in self.split_path.iterdir()
            if folder.is_dir() and folder.name.isdigit()
        ]
        from time import perf_counter

        start = perf_counter()
        self._load_models()
        end = perf_counter()
        print(f"Loading models took {end - start} seconds.")

        self.cache_path = (
            Path.cwd()
            / ".cache"
            / f"dataset_{self.dataset_name}_{self.flag.name.lower()}.pkl"
        )
        if use_cache and self.cache_path.is_file():
            self._load_cache()
            if self._dataset["single_object"] != self.single_object:
                raise ValueError(
                    "Dataset was cached with different single_object flag."
                )
        else:
            self._dataset = self._create_dataset()
            self._dataset_to_single_view() if single_object else None
            self._cache()

    def _load_cache(self) -> None:
        """
        Load the dataset from disk.
        """
        LOGGER.info(f"Loading dataset from {self.cache_path}.")
        try:
            with open(self.cache_path, "rb") as f:
                # self._dataset = json.load(f)
                self._dataset = pickle.load(f)
        except Exception as e:
            LOGGER.info(
                f"Failed to load dataset from {self.cache_path}. Creating dataset from scratch."
            )
            self._dataset = self._create_dataset()
            self._cache()

    def _cache(self) -> None:
        """
        Cache the dataset to disk.
        """
        LOGGER.info(f"Caching dataset to {self.cache_path}.")
        if not self.cache_path.parent.is_dir():
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb+") as f:
            pickle.dump(self._dataset, f)

    def _load_dicts(self, scene_path: str) -> Tuple[Dict[str, Any]]:
        """
        Load ground truth and camera dictionaries from specified scene path.

        Args:
            scene_path (str): Path to the directory containing the scene_gt.json, scene_gt_info.json, and scene_camera.json files.

        Returns:
            A tuple of three dictionaries:
            - gt_dict: A dictionary containing ground truth pose information about the scene.
            - gt_info_dict: A dictionary containing bounding box information of the ground truth objects.
            - cam_dict: A dictionary containing camera information about the scene.
        """
        with open(scene_path / "scene_gt.json", "r") as f:
            gt_dict = json.load(f)
        with open(scene_path / "scene_gt_info.json", "r") as f:
            gt_info_dict = json.load(f)
        with open(scene_path / "scene_camera.json", "r") as f:
            cam_dict = json.load(f)
        return gt_dict, gt_info_dict, cam_dict

    def _dataset_to_single_view(self) -> None:
        """
        Transforms the dataset dictionary to a single view dataset. I.e. each object
        in an image is treated as a separate image.
        """
        raw_img_dataset = self._dataset["raw_img_dataset"].copy()
        single_raw_img_dataset = []
        for entry in tqdm(raw_img_dataset):
            annotation = entry["annotation"]
            for key, value in annotation.items():
                for i in range(len(value)):
                    new_annotation = annotation.copy()
                    new_annotation[key] = [value[i]]
                    entry.update({"annotation": new_annotation})
                    single_raw_img_dataset.append(entry)

        self._dataset["raw_img_dataset"] = single_raw_img_dataset
        self._dataset.update({"single_object": self.single_object})

    def _create_annotation(
        self,
        scene_root: str,
        str_im_id: str,
        gt_dict: Dict[str, Any],
        gt_info_dict: Dict[str, Any],
    ) -> Dict[str, List[Any]]:
        """
        Create an annotation dictionary containing information about a single image.

        Args:
            scene_root: The path to the directory containing the scene information.
            str_im_id: The ID of the image to create the annotation for.
            gt_dict: A dictionary containing ground truth information about the scene.
            gt_info_dict: A dictionary containing additional information about the ground truth objects.

        Returns:
            A dictionary containing the following keys:
            - obj_id: A list of object IDs in the image.
            - bbox_obj: A list of object bounding boxes in the image.
            - bbox_visib: A list of visible bounding boxes in the image.
            - pose: A list of object poses in the image.
            - mask_path: A list of paths to the masks for the objects in the image.
            - mask_visib_path: A list of paths to the visible masks for the objects in the image.
            - graph_path: A list of paths to the XYZ data for the objects in the image.
        """

        annotations = {
            "obj_id": [],
            "bbox_obj": [],
            "bbox_visib": [],
            "pose": [],
            "mask_path": [],
            "mask_visib_path": [],
            "graph_path": [],
        }
        for anno_i, anno in enumerate(gt_dict[str_im_id]):
            obj_id = anno["obj_id"]
            R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
            t = np.array(anno["cam_t_m2c"], dtype="float32")
            pose = np.hstack([R, t.reshape(3, 1)])

            bbox_visib = np.array(
                gt_info_dict[str_im_id][anno_i]["bbox_visib"], dtype=np.int32
            )
            bbox_obj = np.array(
                gt_info_dict[str_im_id][anno_i]["bbox_obj"], dtype=np.int32
            )

            mask_path = str(
                scene_root / "mask" / f"{int(str_im_id):06d}_{int(anno_i):06d}.jpg"
            )
            mask_path = correct_suffix(mask_path)

            mask_visib_path = str(
                scene_root
                / "mask_visib"
                / f"{int(str_im_id):06d}_{int(anno_i):06d}.jpg"
            )
            mask_visib_path = correct_suffix(mask_visib_path)
            graph_path = str(
                scene_root / "graphs" / f"{int(str_im_id):06d}_{int(anno_i):06d}.npz"
            )

            annotations["obj_id"].append(obj_id)
            annotations["bbox_obj"].append(bbox_obj)
            annotations["bbox_visib"].append(bbox_visib)
            annotations["pose"].append(pose)
            annotations["mask_path"].append(mask_path)
            annotations["mask_visib_path"].append(mask_visib_path)
            annotations["graph_path"].append(graph_path)

        # if bbox_visib only zeros return None
        if (
            np.sum(annotations["bbox_obj"][0]) == 0
            or np.sum(annotations["bbox_visib"][0]) == 0
        ):
            return None
        return annotations

    def _create_scene_dataset(self, scene_root: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Create a dataset dictionary from information in a single scene folder.

        Args:
            scene_root: The path to the root directory of the scene.

        Returns:
            A list of dictionaries containing the following keys:
            - scene_id: The ID of the scene.
            - img_id: The ID of the image in the scene.
            - cam: The camera matrix for the image.
            - depth_factor: The depth factor for the image.
            - img_type: The type of image (e.g. real or synthetic).
            - img_path: The path to the image file.
            - annotation: A dictionary containing annotations for the image.
        """
        scene_id = int(str(scene_root.name))
        gt_dict, gt_info_dict, cam_dict = self._load_dicts(scene_root)

        scene_dataset = []
        pbar = gt_dict
        for str_im_id in pbar:
            K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
            depth_scale = cam_dict[str_im_id]["depth_scale"]
            img_path = str(scene_root / "rgb" / "{:06d}.jpg".format(int(str_im_id)))
            img_path = correct_suffix(img_path)
            annotations = self._create_annotation(
                scene_root=scene_root,
                str_im_id=str_im_id,
                gt_dict=gt_dict,
                gt_info_dict=gt_info_dict,
            )
            if not annotations:
                continue
            raw_img_data = {
                "scene_id": scene_id,
                "img_id": int(str_im_id),
                "cam": K,
                "depth_factor": depth_scale,
                "img_type": "real",  # "syn_pbr",
                "img_path": img_path,
                "annotation": annotations,
            }
            scene_dataset.append(raw_img_data)
        return scene_dataset or None

    def _create_dataset(self) -> Dict[str, Any]:
        """
        Generate the dataset dictionary for the current dataset.

        Args:
            root: The path to the root directory of the split of the dataset.

        Returns:
            A dictionary containing the following keys:
            - dataset_name: The name of the dataset.
            - models_info: Information about the models used in the dataset.
            - raw_img_dataset: A list of dictionaries containing information about each image in the dataset
                together with annotations.
        """
        dataset = {
            "dataset_name": self.dataset_name,
            "models_info": self.models_info,
            "raw_img_dataset": [],
        }
        if RANK in {-1, 0}:
            LOGGER.info(f"Creating {self.dataset_name} dataset.")
            pbar = tqdm(self.scene_paths, postfix=f"{self.dataset_name}")
        else:
            pbar = self.scene_paths
        for scene_path in pbar:
            scene_dataset = self._create_scene_dataset(scene_path)
            dataset["raw_img_dataset"].extend(scene_dataset)
        return dataset

    @property
    def models_info(self) -> Dict[str, Any]:
        """
        Load the models_info.json file for the current dataset.

        Returns:
            A dictionary containing information about the models used in the dataset.
        """
        models_info_path = self.models_root / "models_info.json"
        if not models_info_path.exists():
            raise FileNotFoundError("models_info.json not found in models folder")
        with open(models_info_path, "r") as f:
            models_info = json.load(f)
        return models_info

    def _load_models(self) -> None:
        """
        Load the models for the current dataset.
        Format:
        {model_num:int : model:o3d.geometry.TriangleMesh}
        """
        self.models = {
            int(model_num): o3d.io.read_triangle_mesh(
                str(self.models_root / f"obj_{model_num}.ply")
            )
            for model_num in self.model_names
        }

    def generate_graphs(
        self, simplify_factor: int, img_width: int, img_height: int
    ) -> None:
        """
        Generate the graphs for the current dataset.
        """
        if not self.single_object:
            raise NotImplementedError(
                "Graphs are only supported for single object datasets"
            )
        #        if self.graphs_root.exists():
        # LOGGER.info("Graphs already exist. Skipping generation.")
        # return
        # self.graphs_root.mkdir(parents=True, exist_ok=True)
        from data_tools.graph_tools.graph import Graph, prepare_mesh

        LOGGER.info("Generating graphs.")
        dataset = self._dataset["raw_img_dataset"]
        for entry in tqdm(dataset):
            cam = entry["cam"]  # np.ndarray
            annotation = entry["annotation"]
            obj_id = annotation["obj_id"][0]  # int
            pose = annotation["pose"][0]  # np.ndarray
            graph_path = annotation["graph_path"][0]  # str
            if not Path(graph_path).parent.is_dir():
                Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
            model = self.models[obj_id]
            mesh = prepare_mesh(
                mesh=model,
                simplify_factor=simplify_factor,
                pose=pose,
                intrinsic_matrix=cam,
                img_width=img_width,
                img_height=img_height,
            )
            graph = Graph.from_mesh(mesh)
            graph.remove_unconnected_nodes()
            graph.save(graph_path)
        return 0

    #    @property
    #    def models(self) -> np.ndarray:
    #        """
    #        Load the models for the current dataset.
    #
    #        Returns:
    #            A list of dictionaries, each containing information about a single model in the dataset.
    #        """
    #        cache_path = self.models_root / f"models_{self.dataset_name}.pkl"
    #        if self._models is not None:
    #            return self._models
    #        if cache_path.exists() and self.use_cache:
    #            with open(cache_path, "rb") as f:
    #                self._models = pickle.load(f)
    #            return self._models
    #        models = []
    #        for model_num in self.model_names:
    #            model = load_ply(
    #                self.models_root / f"obj_{model_num}.ply",
    #                vertex_scale=self.scale_to_meter,
    #            )
    #            models.append(model)
    #        LOGGER.info(f"cache models to {cache_path}")
    #        with open(cache_path, "wb+") as f:
    #            pickle.dump(models, f)
    #        self._models = models
    #        return models

    @property
    def extents(self) -> np.ndarray:
        """
        Calculate the extents (size) of each model in the dataset.

        Returns:
            An array containing the extents of each model, in the format [size_x, size_y, size_z].
        """
        cur_extents = {}
        models = self.models
        for i, model in enumerate(models):
            pts = model["pts"]
            xmin, xmax = np.amin(pts[:, 0]), np.amax(pts[:, 0])
            ymin, ymax = np.amin(pts[:, 1]), np.amax(pts[:, 1])
            zmin, zmax = np.amin(pts[:, 2]), np.amax(pts[:, 2])
            size_x = xmax - xmin
            size_y = ymax - ymin
            size_z = zmax - zmin
            cur_extents[i] = np.array([size_x, size_y, size_z], dtype="float32")
        return cur_extents

    @require_dataset
    def __len__(self):
        return len(self._dataset["raw_img_dataset"])

    @require_dataset
    def get_obj_ids(self, idx: int) -> List[int]:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["obj_id"]

    @require_dataset
    def get_mask_paths(self, idx: int) -> str:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["mask_path"]

    @require_dataset
    def get_mask_visib_paths(self, idx: int) -> str:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["mask_visib_path"]

    @require_dataset
    def length(self):
        return len(self._dataset["raw_img_dataset"])

    @require_dataset
    def get_img_path(self, idx: int) -> str:
        return self._dataset["raw_img_dataset"][idx]["img_path"]

    @require_dataset
    def get_img_id(self, idx: int) -> int:
        return self._dataset["raw_img_dataset"][idx]["img_id"]

    @require_dataset
    def get_scene_id(self, idx: int) -> int:
        return self._dataset["raw_img_dataset"][idx]["scene_id"]

    @require_dataset
    def get_cam(self, idx: int) -> np.ndarray:
        return self._dataset["raw_img_dataset"][idx]["cam"].copy()

    @require_dataset
    def get_depth_factor(self, idx: int) -> float:
        return self._dataset["raw_img_dataset"][idx]["depth_factor"]

    @require_dataset
    def get_img_type(self, idx: int) -> str:
        return self._dataset["raw_img_dataset"][idx]["img_type"]

    @require_dataset
    def get_obj_ids(self, idx: int) -> int:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["obj_id"]

    @require_dataset
    def get_bbox_objs(self, idx: int) -> List[List[int]]:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["bbox_obj"].copy()

    @require_dataset
    def get_bbox_visibs(self, idx: int) -> np.ndarray:
        return self._dataset["raw_img_dataset"][idx]["annotation"]["bbox_visib"].copy()

    @require_dataset
    def get_poses(self, idx: int) -> np.ndarray:
        return copy.deepcopy(
            self._dataset["raw_img_dataset"][idx]["annotation"]["pose"]
        )
    
    @require_dataset
    def get_graph_paths(self, idx:int) -> Graph:
       return self._dataset["raw_img_dataset"][idx]["annotation"]["graph_path"]
        


def load_ply(path, vertex_scale=1.0):
    # https://github.com/thodan/sixd_toolkit/blob/master/pysixd/inout.py
    # bop_toolkit
    """Loads a 3D mesh model from a PLY file.

    :param path: Path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    -' pts' (nx3 ndarray),
    - 'normals' (nx3 ndarray), optional
    - 'colors' (nx3 ndarray), optional
    - 'faces' (mx3 ndarray), optional.
    - 'texture_uv' (nx2 ndarray), optional
    - 'texture_uv_face' (mx6 ndarray), optional
    - 'texture_file' (string), optional
    """

    f = open(path, "r")

    # Only triangular faces are supported.
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


class LinemodDataset(Dataset):
    def keypoints_to_map(self, mask, pts2d, unit_vectors=True):
        # based on: https://github.com/zju3dv/pvnet/blob/master/lib/datasets/linemod_dataset.py
        mask = mask[0]
        h, w = mask.shape
        n_pts = pts2d.shape[0]
        xy = np.argwhere(mask == 1.0)[:, [1, 0]]
        xy = np.expand_dims(xy.transpose(0, 1), axis=1)
        pts_map = np.tile(xy, (1, n_pts, 1))
        pts_map = (
            np.tile(np.expand_dims(pts2d, axis=0), (pts_map.shape[0], 1, 1)) - pts_map
        )
        if unit_vectors:
            norm = np.linalg.norm(pts_map, axis=2, keepdims=True)
            norm[norm < 1e-3] += 1e-3
            pts_map = pts_map / norm
        pts_map_out = np.zeros((h, w, n_pts, 2), np.float32)
        pts_map_out[xy[:, 0, 1], xy[:, 0, 0]] = pts_map
        pts_map_out = np.reshape(pts_map_out, (h, w, n_pts * 2))
        pts_map_out = np.transpose(pts_map_out, (2, 0, 1))
        return pts_map_out

    def __getitem__(self, idx: int) -> Any:
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

    def __getitem__(self, idx: int) -> Any:
        # return roi image, roi mask, keypoints, bbox, R, t
        img = np.random.rand(self.imgsz, self.imgsz, 3)
        # img = torch.randn(3, self.imgsz, self.imgsz)
        mask = np.random.rand(self.imgsz, self.imgsz)
        mask = (mask > 0.5).astype(np.uint8)
        keypoints = np.random.rand(10, 2)
        bbox = np.random.rand(4)
        class_id = np.array(np.random.randint(0, self.num_classes))
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
        return img, mask, keypoints, bbox, class_id, R, t


dummy_dataset = DummyDataset()
dummy_dataloader = DataLoader(dummy_dataset, batch_size=2, shuffle=True, num_workers=0)

# whole image. set detach graph of weights where attention is below threshold