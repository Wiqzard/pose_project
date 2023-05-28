from __future__ import annotations

from typing import Tuple, List, Optional
import enum
import itertools
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
import open3d as o3d
import numpy as np

TriangleMesh = o3d.geometry.TriangleMesh


class SimplifyMode(enum.Enum):
    """Simplification mode for graph simplification"""

    QUADRATIC = enum.auto()
    VERTEX = enum.auto()


# TODO: seperate feature matrix to X and feature matrix, where
#       the feature matrix has the classification and tracker of each node.
@dataclass
class Graph:
    """
    A data class representing a graph, with adjacency and feature matrices.

    Attributes:
        adjacency_matrix (np.ndarray): A square matrix representing the adjacency between nodes in the graph.
                                    The value at adjacency_matrix[i, j] represents the weight of the edge
                                    between node i and node j. A value of 0 indicates no edge between nodes.
        feature_matrix (np.ndarray): A matrix representing the features of each node in the graph.
                                    Each row of the matrix corresponds to a node, and the columns represent
                                    the feature values of that node.
    """

    adjacency_matrix: np.ndarray
    feature_matrix: np.ndarray

    def __post_init__(self):
        if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if self.adjacency_matrix.shape[0] != self.feature_matrix.shape[0]:
            raise ValueError(
                "Feature matrix must have same number of rows as adjacency matrix"
            )
        if self.adjacency_matrix is not None:
            self.adjacency_matrix = self.adjacency_matrix.astype(np.float32)
        if self.feature_matrix is not None:
            self.feature_matrix = self.feature_matrix.astype(np.float32)

    def __len__(self) -> int:
        """
        Returns:
            int: Number of nodes in the graph
        """
        return self.feature_matrix.shape[0]

    @property
    def num_nodes(self) -> int:
        """
        Returns:
            int: Number of nodes in the graph
        """
        return self.feature_matrix.shape[0]

    @property
    def num_edges(self) -> int:
        """
        Returns:
            int: Number of edges in the graph
        """
        return int(np.sum(self.adjacency_matrix) / 2)

    @property
    def num_features(self) -> int:
        """
        Returns:
            int: Number of features per node in the graph
        """
        return self.feature_matrix.shape[1]

    @property
    def degree_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: A diagonal matrix where the value at node_degree_matrix[i, i] is the degree of node i
        """
        return np.diag(np.sum(self.adjacency_matrix, axis=1))

    @property
    def normed_degree_matrix(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: A diagonal matrix where the value at normed_degree_matrix[i, i] is the normed degree of node i
        """
        return np.diag(self.degree_matrix**-0.5)

    def remove_unconnected_nodes(self) -> None:
        """
        Remove unconnected nodes from the adjacency and feature matrices of the graph.

        This method finds nodes with no connections (degree of 0) and removes them from both the adjacency matrix
        and the feature matrix, effectively eliminating these nodes from the graph.
        """
        unconnected_nodes = np.where(np.sum(self.adjacency_matrix, axis=1) == 0)[0]
        adjacency_matrix = np.delete(self.adjacency_matrix, unconnected_nodes, axis=0)
        self.adjacency_matrix = np.delete(adjacency_matrix, unconnected_nodes, axis=1)
        self.feature_matrix = np.delete(self.feature_matrix, unconnected_nodes, axis=0)

    def transform_features_to_site(
        self, cam_k: np.ndarray, im_w: int, im_h: int, scale: Optional[int] = None
    ) -> None:
        """
        Transforms the features of the graph to site coordinates.

        This method transforms the features of the graph from object coordinates to site coordinates.
        """
        coords_3d = self.feature_matrix[:, :3]
        coords_2d = coords_3d @ cam_k.T
        coords_2d[:, :2] /= coords_2d[:, 2][:, None] * np.array((im_w, im_h)).reshape(
            1, 2
        )
        self.feature_matrix[:, :2] = coords_2d[:, :2].astype(np.float32)

    def transform_features_to_3d_coords(self, cam_k: np.ndarray, im_w: int, im_h: int):
        """
        Transforms the features of the graph to 3D coordinates.

        This method transforms the features of the graph from site coordinates to object coordinates.
        """
        coords_3d = self.feature_matrix[:, :3]
        coords_3d[:, :2] *= coords_3d[:, 2][:, None] * np.array((im_w, im_h)).reshape(
            1, 2
        )
        coords_3d = np.linalg.inv(cam_k) @ coords_3d.T
        self.feature_matrix[:, :3] = coords_3d.T.astype(np.float32)

    def add_self_loop(self) -> None:
        """
        Add self loop to the graph.

        This method adds a self loop to the graph, by adding 1 to the diagonal of the adjacency matrix.
        """
        self.adjacency_matrix += np.eye(self.num_nodes)

    def set_adjacency_list(self) -> None:
        """
        Set the adjacency list of the graph.

        This method sets the adjacency list of the graph, by converting the adjacency matrix to a list of lists.
        """
        self.adjacency_list = [
            np.where(self.adjacency_matrix[i] == 1)[0] for i in range(self.num_nodes)
        ]

    def set_edge_index(self) -> None:
        """
        Set the edge index list of the graph.

        This method sets the edge index list of the graph, by converting the adjacency matrix to a list of tuples.
        """
        self.edge_index = np.array(
            [
                (i, j)
                for i in range(self.num_nodes)
                for j in np.where(self.adjacency_matrix[i] == 1)[0]
            ]
        )

    @classmethod
    def create_random_graph(cls, num_nodes: int, num_features: int) -> Graph:
        """
        Create a random graph.

        Args:
            num_nodes (int): Number of nodes in the graph.
            num_features (int): Number of features per node in the graph.

        Returns:
            Graph: A random graph.
        """
        adjacency_matrix = np.random.randint(0, 2, size=(num_nodes, num_nodes))
        adjacency_matrix = np.triu(adjacency_matrix, k=1)
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T
        feature_matrix = np.random.rand(num_nodes, num_features)
        return cls(adjacency_matrix=adjacency_matrix, feature_matrix=feature_matrix)

    @classmethod
    def create_initial_graph(cls, num_nodes: int, num_features: int) -> Graph:
        """
        Create a initial graph. Every node is connected to 3 other nodes.

        Args:
            num_nodes (int): Number of nodes in the graph.
            num_features (int): Number of features per node in the graph.

        Returns:
            Graph: A initial graph.
        """
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        for i, j in itertools.product(range(num_nodes), range(1, 4)):
            adjacency_matrix[i, (i + j) % num_nodes] = 1
            adjacency_matrix[(i + j) % num_nodes, i] = 1
        feature_matrix = np.random.rand(num_nodes, num_features)
        return cls(adjacency_matrix=adjacency_matrix, feature_matrix=feature_matrix)

    @staticmethod
    def to_torch_geometric(graph: Graph) -> Data:
        """
        Convert a graph to a PyTorch Geometric Data object.

        Args:
            graph (Graph): The graph to convert.

        Returns:
            Data: The converted graph.
        """
        if getattr(graph, "edge_index_list", None) is None:
            graph.set_edge_index_list()
        return Data(
            x=torch.tensor(graph.feature_matrix),
            edge_index=torch.tensor(graph.edge_index_list).T,
        )

    def save(self, path: str | Path) -> None:
        """
        Save the graph to disk.

        Args:
            path (str | Path): Path to save the graph to.
        """
        np.savez(
            str(path),
            adjacency_matrix=self.adjacency_matrix,
            feature_matrix=self.feature_matrix,
        )

    @classmethod
    def load(cls, path: str | Path) -> Graph:
        """
        Load a graph from diska, concat=True, negative_slope=0.2, dropout=0.5)

        Args:
            path (str): Path to load the graph from.

        Returns:
            Graph: The loaded graph.
        """
        data = np.load(str(path))
        return cls(
            adjacency_matrix=data["adjacency_matrix"],
            feature_matrix=data["feature_matrix"],
        )

    @classmethod
    def from_mesh(
        cls,
        mesh: o3d.geometry.TriangleMesh,
        simplify_mode: Optional[SimplifyMode] = None,
        **kwargs,
    ) -> Graph:
        """
        Create a Graph object from a TriangleMesh, with an optional simplification step.

        Args:
            mesh (o3d.geometry.TriangleMesh): The input TriangleMesh to be converted into a graph representation.
            simplify_mode (Optional[SimplifyMode]): The simplification method to be applied to the input mesh.
                                                Options are SimplifyMode.QUADRATIC or SimplifyMode.VERTEX.
                                                If None, no simplification is applied. Default is None.
            **kwargs: Keyword arguments passed to the simplification method.

        Returns:
            Graph: A Graph object representing the input TriangleMesh.
        """
        if simplify_mode == SimplifyMode.QUADRATIC:
            mesh.simplify_quadric_decimation(kwargs)
        elif simplify_mode == SimplifyMode.VERTEX:
            mesh.simplify_vertex_clustering(kwargs)
        adjacency_matrix, feature_matrix = mesh_to_graph(mesh)
        return cls(adjacency_matrix, feature_matrix)

    def visualize(self) -> None:
        """
        Visualize the graph representation of the object in 3D space using the adjacency matrix and feature matrix.

        This method plots the nodes as blue points and edges as grey lines in a 3D scatter plot. Node positions are
        determined by the feature matrix. The plot is displayed using the matplotlib library.
        """
        G = nx.from_numpy_array(self.adjacency_matrix)
        node_positions = {node: tuple(self.feature_matrix[node]) for node in G.nodes()}
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for position in node_positions.values():
            ax.scatter(*position, s=10, c="blue", alpha=0.9)
        for edge in G.edges():
            x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
            y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
            z_coords = [node_positions[edge[0]][2], node_positions[edge[1]][2]]
            ax.plot(x_coords, y_coords, z_coords, color="grey", alpha=0.9)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        plt.show()


def adjacency_list_to_matrix(adjacency_list: List[set]) -> np.ndarray:
    """
    Converts an adjacency list representation of a graph to its equivalent adjacency matrix representation.

    Args:
    adjacency_list (list): A list of sets, where each inner list represents the neighbors of a vertex in the graph.

    Returns:
    np.ndarray: A 2D numpy array representing the adjacency matrix of the graph. The element at (i,j) is 1 if there is an
    edge from vertex i to vertex j in the graph, and 0 otherwise.
    """
    num_vertices = len(adjacency_list)
    adjacency_matrix = np.zeros((num_vertices, num_vertices))

    for i, neighbors in enumerate(adjacency_list):
        for neighbor in neighbors:
            adjacency_matrix[i, neighbor] = 1

    return adjacency_matrix


def mesh_to_graph(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a 3D triangle mesh to its equivalent graph representation.

    Args:
    mesh (o3d.geometry.TriangleMesh): A 3D triangle mesh object to be converted to graph representation.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays representing the adjacency matrix and the feature matrix of
    the graph. The adjacency matrix is a 2D numpy array where the element at (i,j) is 1 if there is an edge between vertices
    i and j in the graph, and 0 otherwise. The feature matrix is a 2D numpy array where each row represents a vertex of the
    graph, and each column represents a feature of the vertex. (m)
    """
    adjacency_list = mesh.compute_adjacency_list().adjacency_list
    adjacency_matrix = adjacency_list_to_matrix(adjacency_list)
    feature_matrix = np.asarray(mesh.vertices) / 1000
    return adjacency_matrix, feature_matrix


def prepare_mesh(
    mesh: TriangleMesh,
    simplify_factor: float,
    pose: np.ndarray,
    intrinsic_matrix: np.ndarray,
    img_width: int,
    img_height: int,
) -> TriangleMesh:
    """
    Prepare a mesh for rendering by simplifying its geometry, applying a pose transformation, and removing
    invisible vertices.

    Args:
        mesh (TriangleMesh): The input mesh to be processed.
        simplify_factor (float): The simplification factor for vertex clustering, between 0 and 1. Higher values result
                                 in greater simplification.
        pose (np.ndarray): A 3x4 pose matrix representing the transformation to be applied to the mesh.
        intrinsic_matrix (np.ndarray): A 3x3 intrinsic camera matrix.
        img_width (int): The width of the image frame.
        img_height (int): The height of the image frame.

    Returns:
        TriangleMesh: The processed mesh with simplified geometry, transformed pose, and invisible vertices removed.
    """
    mesh = mesh.simplify_vertex_clustering(simplify_factor)
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :] = pose
    mesh = mesh.transform(transformation_matrix)
    mesh = remove_invisible_vertices(
        mesh, pose, intrinsic_matrix, img_width, img_height
    )
    return mesh


def project_points(vertices: np.ndarray, intrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    Project 3D vertices to 2D image space using an intrinsic camera matrix.

    Args:
        vertices (np.ndarray): An array of shape (N, 3) representing the 3D vertices.
        intrinsic_matrix (np.ndarray): A 3x3 intrinsic camera matrix.

    Returns:
        np.ndarray: An array of shape (N, 2) representing the 2D projected points.
    """
    projected_points = intrinsic_matrix @ vertices.T
    projected_points /= projected_points[2]
    return projected_points[:2].T


def visible_vertices_mask(
    vertices: np.ndarray, img_width: int, img_height: int
) -> np.ndarray:
    """
    Determine the visibility of vertices based on their position in the image frame.

    Args:
        vertices (np.ndarray): An array of shape (N, 2) representing the 2D vertices in image space.
        img_width (int): The width of the image frame.
        img_height (int): The height of the image frame.

    Returns:
        np.ndarray: A boolean array of shape (N,) indicating whether each vertex is visible or not.
    """
    mask_x = (vertices[:, 0] >= 0) & (vertices[:, 0] < img_width)
    mask_y = (vertices[:, 1] >= 0) & (vertices[:, 1] < img_height)
    return mask_x & mask_y


def remove_hidden_vertices(
    vertices: np.ndarray, faces: np.ndarray, model_view_matrix: np.ndarray
) -> np.ndarray:
    """
    Remove hidden faces from a mesh given its vertices, faces, and the model-view transformation matrix.
    This function assumes backface culling, i.e., faces not visible from the camera position are removed.

    Args:
        vertices (np.ndarray): An array of shape (N, 3) representing the vertices of the mesh.
        faces (np.ndarray): An array of shape (M, 3) representing the faces of the mesh, defined by vertex indices.
        model_view_matrix (np.ndarray): A 4x4 model-view transformation matrix.

    Returns:
        np.ndarray: An array of shape (M', 3) representing the visible faces after removing hidden ones.
    """
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    camera_position = (
        -np.linalg.inv(model_view_matrix)[:3, :3].T @ model_view_matrix[:3, 3]
    )
    visible_faces_mask = np.dot(face_normals, camera_position) > 0
    return faces[visible_faces_mask]


def remove_invisible_vertices(
    mesh: TriangleMesh,
    pose: np.ndarray,
    intrinsic_matrix: np.ndarray,
    img_width: int,
    img_height: int,
) -> TriangleMesh:
    """
    Remove invisible vertices from a mesh given the camera pose, intrinsic matrix, and image dimensions.

    Args:
        mesh (TriangleMesh): The input mesh from which to remove invisible vertices.
        pose (np.ndarray): A 3x4 pose matrix representing the transformation to be applied to the mesh.
        intrinsic_matrix (np.ndarray): A 3x3 intrinsic camera matrix.
        img_width (int): The width of the image frame.
        img_height (int): The height of the image frame.

    Returns:
        TriangleMesh: A new mesh with only the visible vertices and corresponding faces.
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :] = pose
    projected_points = project_points(vertices, intrinsic_matrix)
    visible_mask = visible_vertices_mask(projected_points, img_width, img_height)
    # visible_vertices = vertices[visible_mask]

    visible_vertices = vertices

    visible_mask_indices = np.where(~visible_mask)[0]

    visible_faces = remove_hidden_vertices(
        visible_vertices, faces, transformation_matrix
    )

    visible_vertices = visible_vertices[visible_mask]

    visible_faces = np.asarray(
        [
            face
            for face in visible_faces
            if not np.any(np.isin(visible_faces, visible_mask_indices))
        ]
    )

    new_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(visible_vertices),
        o3d.utility.Vector3iVector(visible_faces),
    )
    return new_mesh
