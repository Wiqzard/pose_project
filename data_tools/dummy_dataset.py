from typing import Tuple, Union, Any
from enum import Enum, auto

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import open3d as o3d

from data_tools.bop_dataset import BOPDataset, Flag, DatasetType
from data_tools.graph_tools.graph import Graph, prepare_mesh
from utils.bbox import Bbox



class InitialModel(Enum):
    TETRAHEDRON = auto()
    SPHERE = auto()


class DummyDataset(Dataset):
    ARROW = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=10, cone_radius=20, cylinder_height=70, cone_height=50, resolution=10, cylinder_split=4, cone_split=1)

    def __init__(self, bop_dataset: BOPDataset, cfg=None) -> None:
        super().__init__()
        self.dataset = bop_dataset
        self.length = 0.1
        self.im_height, self.im_width  = 480, 640
        

    def __len__(self) -> int:
        return len(self.dataset)

    def _generate_initial_graph(self, idx: int) -> Graph:
        bbox_objs = self.dataset.get_bbox_objs(idx)
        # could be cx cy h w, byt also x y h w (left top)
        centers = np.array([np.asarray([bbox[0]/self.im_width,bbox[1]/self.im_height]) for bbox in bbox_objs])
        initial_feat, initial_adj = self.get_inital_model(
            self.length, InitialModel.TETRAHEDRON
        )

        # shift the nodes by the centers and z to 0.5
        initial_features = np.tile(initial_feat, (centers.shape[0], 1))
        centers_adj = np.repeat(centers, initial_feat.shape[0], axis=0)
        initial_features[:, :2] += centers_adj
        initial_features[:, 2] += 0.5
        
        num_nodes = centers.shape[0] * 4
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(centers.shape[0]):
            adjacency_matrix[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = initial_adj[
                :4, :4
            ]
        graph = Graph(
            adjacency_matrix=adjacency_matrix, feature_matrix=initial_features
        )
        graph.set_edge_index()
        return graph

    def _generate_graph_for_img(self, idx: int) -> Graph:
        poses = self.dataset.get_poses(idx)
        cam = self.dataset.get_cam(idx)
        num_objs = len(poses)
        total_num_nodes = num_objs * 64
        feature_matrix = np.zeros((total_num_nodes, 3))
        adjacency_matrix = np.zeros((total_num_nodes, total_num_nodes)) 

        for i in range(num_objs):
            mesh = prepare_mesh(mesh=self.ARROW, simplify_factor=0, pose=poses[i], intrinsic_matrix=cam,
                                img_width=self.im_width, img_height=self.im_height)
            graph = Graph.from_mesh(mesh)
            adjacency_matrix[i * 64 : (i + 1) * 64, i * 64 : (i + 1) * 64] = graph.adjacency_matrix 
            features = graph.feature_matrix - np.mean(graph.feature_matrix, axis=0)
            feature_matrix[i * 64 : (i + 1) * 64, :] = features
        graph = Graph(
            adjacency_matrix=adjacency_matrix, feature_matrix=feature_matrix
        )
        graph.set_edge_index()
        return graph
         
    def get_inital_model(
        self, length: float, model_type: InitialModel
    ) -> Tuple[np.ndarray, np.ndarray]:
        if model_type == InitialModel.TETRAHEDRON:
            tetrahedron = np.array(
                [
                    [0, 0, 0],
                    [length, 0, 0],
                    [length / 2, length * np.sqrt(3) / 2, 0],
                    [length / 2, length / (2 * np.sqrt(3)), length * np.sqrt(2 / 3)],
                ]
            )
            tetra_adjacency_matrix = np.array(
                [
                    [0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [1, 1, 0, 1],
                    [1, 1, 1, 0],
                ]
            )
            tetrahedron -= np.mean(tetrahedron, axis=0)
            return tetrahedron, tetra_adjacency_matrix



def edge_based_unpooling(edge_index, feature_matrix):
    num_nodes = feature_matrix.shape[0]
    num_edges = edge_index.shape[0]

    # Calculate new node coordinates
    new_node_coords = np.mean(feature_matrix[edge_index], axis=1)

    # Append new node coordinates to feature matrix
    new_feature_matrix = np.vstack([feature_matrix, new_node_coords])

    # Create new edge index with new edges connecting old nodes with new nodes
    new_node_indices = np.arange(num_nodes, num_nodes + num_edges)[:, None]
    new_edges_1 = np.hstack([edge_index[:, 0].reshape(-1, 1), new_node_indices])
    new_edges_2 = np.hstack([new_node_indices, edge_index[:, 1].reshape(-1, 1)])

    # Initialize new edge index with old edges and new edges
    new_edge_index = np.vstack([edge_index, new_edges_1, new_edges_2])

    # Connect the three new vertices for each old triangle
    for i in range(num_nodes, num_nodes + num_edges, 3):
        new_edge_index = np.vstack(
            [new_edge_index, [[i, i + 1], [i + 1, i + 2], [i + 2, i]]]
        )

    return new_edge_index, new_feature_matrix


# def edge_based_unpooling(edge_index, feature_matrix):
#    """dont forget to remove the additional edge placeholders"""
#    num_nodes = feature_matrix.shape[0]
#    num_edges = edge_index.shape[0]
#
#    # Initialize new feature matrix by copying old matrix and adding space for new nodes
#    new_feature_matrix = np.zeros((num_nodes + num_edges, 3))
#    new_feature_matrix[:num_nodes] = feature_matrix
#
#    # Initialize new edge index by copying old edge index and adding space for new edges
#    new_edge_index = np.zeros((num_edges * 2 + num_nodes * 3 * 10, 2), dtype=int)
#    new_edge_index[:num_edges] = edge_index
#
#    # Process each edge
#    for i, edge in enumerate(edge_index):
#        # Calculate new node coordinate
#        new_node_coord = (feature_matrix[edge[0]] + feature_matrix[edge[1]]) / 2
#        new_node_index = num_nodes + i
#
#        # Update new feature matrix
#        new_feature_matrix[new_node_index] = new_node_coord
#
#        # Update new edge index
#        new_edge_index[num_edges + i*2] = [edge[0], new_node_index]
#        new_edge_index[num_edges + i*2 + 1] = [new_node_index, edge[1]]
#
#    # Connect the three new vertices for each old triangle
#    for i in range(num_nodes, num_nodes + num_edges, 3):
#        new_edge_index[num_edges * 2 + (i - num_nodes)] = [i, i + 1]
#        new_edge_index[num_edges * 2 + (i - num_nodes) + 1] = [i + 1, i + 2]
#        new_edge_index[num_edges * 2 + (i - num_nodes) + 2] = [i + 2, i]
#
#    return new_edge_index, new_feature_matrix
