from __future__ import annotations

from typing import Tuple, List, Optional
import enum
import math
from dataclasses import dataclass

import open3d as o3d
import numpy as np


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
    graph, and each column represents a feature of the vertex.
    """
    adjacency_list = mesh.compute_adjacency_list().adjacency_list
    adjacency_matrix = adjacency_list_to_matrix(adjacency_list)
    feature_matrix = np.asarray(mesh.vertices)
    return adjacency_matrix, feature_matrix


# Visualize the original and simplified meshes
def visualize_mesh(mesh: o3d.geometry.TriangleMesh) -> None:
    mesh.paint_uniform_color([0.2, 0.5, 0.5])  # RGB values in the range [0, 1]
    # RGB values in the range [0, 1]
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name="Original Mesh")
    # Save the simplified mesh as a new PLY file
    # o3d.io.write_triangle_mesh('your_simplified_object.ply', simplified_mesh)


# Visualize Graph
def visualize_graph(adjacency_matrix: np.ndarray, feature_matrix: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.from_numpy_array(adjacency_matrix)
    node_positions = {node: tuple(feature_matrix[node]) for node in G.nodes()}
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # Draw the nodes
    for position in node_positions.values():
        ax.scatter(*position, s=10, c="blue", alpha=0.9)
    # Draw the edges
    for edge in G.edges():
        x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
        y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
        z_coords = [node_positions[edge[0]][2], node_positions[edge[1]][2]]
        ax.plot(x_coords, y_coords, z_coords, color="grey", alpha=0.9)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()


def axangle2mat(axis, angle, is_normalized=False):
    """Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis : 3 element sequence
       vector specifying axis for rotation.
    angle : scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.
    Returns
    -------
    mat : array shape (3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    x, y, z = axis
    if not is_normalized:
        n = math.sqrt(x * x + y * y + z * z)
        x = x / n
        y = y / n
        z = z / n
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC
    return np.array(
        [
            [x * xC + c, xyC - zs, zxC + ys],
            [xyC + zs, y * yC + c, yzC - xs],
            [zxC - ys, yzC + xs, z * zC + c],
        ]
    )


def allocentric_to_egocentric(allo_pose, cam_ray=(0, 0, 1.0)):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.
    Since objects further away from the optical center undergo skewing,
    we try to visually correct by rotating back the amount between
    optical center ray and object centroid ray. Another way to solve
    that might be translational variance
    (https://arxiv.org/abs/1807.03247)
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    trans = allo_pose[:3, 3]
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount

    if angle > 0:
        ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
        ego_pose[:3, 3] = trans
        rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
        ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
    else:
        ego_pose = allo_pose.copy()
    return ego_pose


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

    def __len__(self):
        return self.feature_matrix.shape[0]

    def simplify_graph(self) -> None:
        pass

    def transform_to_camera_frame(self, pose: np.ndarray) -> None:
        """Pose must be a 4x4 matrix, (rot, trans) trans in meters"""
        if pose.shape != (4, 4):
            raise ValueError("Pose must be a 4x4 matrix")
        self.feature_matrix = np.dot(self.feature_matrix, pose[:3, :3].T) + pose[:3, 3]

    def visualize(self):
        pass

    def save(self, path: str) -> None:
        pass

    def remove_unconnected_nodes(self) -> None:
        unconnected_nodes = np.where(np.sum(self.adjacency_matrix, axis=1) == 0)[0]
        adjacency_matrix = np.delete(self.adjacency_matrix, unconnected_nodes, axis=0)
        self.adjacency_matrix = np.delete(adjacency_matrix, unconnected_nodes, axis=1)
        self.feature_matrix = np.delete(self.feature_matrix, unconnected_nodes, axis=0)

    @classmethod
    def load(cls, path: str) -> Graph:
        pass

    @classmethod
    def from_mesh(
        cls,
        mesh: o3d.geometry.TriangleMesh,
        simplify_mode: Optional[SimplifyMode] = None,
        **kwargs,
    ) -> Graph:
        if simplify_mode == SimplifyMode.QUADRATIC:
            mesh.simplify_quadric_decimation(kwargs)
        elif simplify_mode == SimplifyMode.VERTEX:
            mesh.simplify_vertex_clustering(kwargs)

        adjacency_matrix, feature_matrix = mesh_to_graph(mesh)
        return cls(adjacency_matrix, feature_matrix)


# get visible faces


def get_visible_vertices(mesh, cam_intrinsic) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(mesh)
    ctr = vis.get_view_control()

    # retrieve intrinsic camera settings
    parameters = o3d.io.read_pinhole_camera_parameters(cam_intrinsic)
    ctr.convert_from_pinhole_camera_parameters(parameters)
    depth = vis.capture_depth_float_buffer(False)
    image = vis.capture_screen_float_buffer(False)
    vis.run()
    vis.destroy_window()
    return depth, image
    # visualizer.destroy_window()


class SimplifyMode(enum.Enum):
    """Simplification mode for graph simplification"""

    QUADRATIC = enum.auto()
    VERTEX = enum.auto()


def project_points(vertices, model_view_matrix, intrinsic_matrix):
    homogeneous_vertices = np.column_stack((vertices, np.ones(vertices.shape[0])))
    projected_points = intrinsic_matrix @ model_view_matrix @ homogeneous_vertices.T
    projected_points /= projected_points[2]
    return projected_points[:2].T


def visible_vertices_mask(vertices, img_width, img_height):
    mask_x = (vertices[:, 0] >= 0) & (vertices[:, 0] < img_width)
    mask_y = (vertices[:, 1] >= 0) & (vertices[:, 1] < img_height)
    return mask_x & mask_y


def remove_hidden_vertices(vertices, faces, model_view_matrix):
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


def remove_invisible_vertices(mesh, pose, intrinsic_matrix, img_width, img_height):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :] = pose
    # model_view_matrix = np.linalg.inv(transformation_matrix)
    projected_points = project_points(vertices, pose, intrinsic_matrix)
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


def main() -> int:
    # Load the PLY file
    mesh = o3d.io.read_triangle_mesh(
        "/Users/sebastian/Documents/Projects/pose_project/data/datasets/obj_000006.ply"
    )
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    mesh = mesh.simplify_vertex_clustering(10.1)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])
    graph = Graph.from_mesh(mesh)
    pose = np.array(
        [
            [0.0950661, 0.98330897, -0.155129, 71.62781422],
            [0.74159598, -0.173913, -0.64791101, -158.20064191],
            [-0.66407597, -0.0534489, -0.74575198, 1050.77777823],
        ]
    )
    cam_K = np.array(
        [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]
    ).reshape(3, 3)
    # Transform the mesh to the camera frame
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :] = pose
    # transformation_matrix[:3, 3] *= 1000
    # transformation_matrix[3, 3] *= -1 # we need this for the correct vertices, but projected points get wrong
    # mesh.transform(transformation_matrix)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)  # every vertex in in mm checks
    triangles = np.asarray(mesh.triangles)
    # visualize mesh
    new_mewsh = remove_invisible_vertices(
        mesh=mesh,
        pose=pose,
        intrinsic_matrix=cam_K,
        img_width=640,
        img_height=480,
    )

    new_mewsh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([new_mewsh])
    graph = Graph.from_mesh(new_mewsh)
    print(len(graph))
    graph.remove_unconnected_nodes()
    print(len(graph))
    visualize_graph(graph.adjacency_matrix, graph.feature_matrix)

    def project_points(vertices, intrinsic_matrix):
        # homogeneous_vertices = np.column_stack((vertices, np.ones(vertices.shape[0])))
        projected_points = intrinsic_matrix @ vertices.T  # homogeneous_vertices.T
        projected_points /= projected_points[2]
        return projected_points[:2].T

    p = project_points(np.asarray(mesh.vertices), cam_K)

    # Define camera parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=640, height=480, fx=572.4114, fy=573.57043, cx=325.2611, cy=242.04899
    )
    image, depth = get_visible_vertices(mesh, intrinsic)

    vertex_normals = np.asarray(mesh.vertex_normals)
    cam_vector = np.array([0, 0, 1])
    # filter out the ones that are not visible

    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic = intrinsic

    graph.transform_to_camera_frame(transformation_matrix)

    # get the ones that are in front of the camera visible

    # visualize_graph(graph.adjacency_matrix, graph.feature_matrix)

    return 0


if __name__ == "__main__":
    main()

# mesh.transform(transformation_matrix)
#
## Adjacency list
# adjacency_list = mesh.compute_adjacency_list().adjacency_list
# adjacency_list_simplified = simplified_mesh.compute_adjacency_list().adjacency_list
# adjacency_list_simplified_vertex = (
#    simplified_mesh_vertex.compute_adjacency_list().adjacency_list
# )
# adjacency_matrix = adjacency_list_to_matrix(adjacency_list)
# adjacency_matrix_vertex = adjacency_list_to_matrix(adjacency_list_simplified_vertex)
#
# feature_matrix = np.asarray(mesh.vertices)
# feature_matrix_vertex = np.asarray(simplified_mesh_vertex.vertices)
##

# Reduce vertices
# target_num_triangles = 1000  # Set your desired number of vertices
# simplified_mesh = mesh.simplify_quadric_decimation(target_num_triangles)
#    cam_K = np.array(
#        [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0]
#    ).reshape(3, 3)
#    rot = np.array(
#        [
#            0.0963063,
#            0.99404401,
#            0.0510079,
#            0.57332098,
#            -0.0135081,
#            -0.81922001,
#            -0.81365103,
#            0.10814,
#            -0.57120699,
#        ]
#    ).reshape(3, 3)
#    trans = np.array([-105.3577515, -117.52119142, 1014.8770132]).reshape(3, 1) / 1000
#    pose = np.concatenate((rot, trans), axis=1)
