from __future__ import annotations

from typing import Tuple, List, Optional
import enum
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import open3d as o3d
import numpy as np

TriangleMesh = o3d.geometry.TriangleMesh


class SimplifyMode(enum.Enum):
    """Simplification mode for graph simplification"""

    QUADRATIC = enum.auto()
    VERTEX = enum.auto()


@dataclass
class Graph:
    adjacency_matrix: np.ndarray
    feature_matrix: np.ndarray

    def __post_init__(self):
        if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square")
        if self.adjacency_matrix.shape[0] != self.feature_matrix.shape[0]:
            raise ValueError(
                "Feature matrix must have same number of rows as adjacency matrix"
            )

    def __len__(self) -> int:
        """
        Returns:
            int: Number of nodes in the graph
        """
        return self.feature_matrix.shape[0]

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
        Load a graph from disk.

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
    visible_vertices = vertices[visible_mask]
    visible_faces = remove_hidden_vertices(
        visible_vertices, faces, transformation_matrix
    )

    new_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(visible_vertices),
        o3d.utility.Vector3iVector(visible_faces),
    )
    return new_mesh
