import numpy as np
import torch
import open3d as o3d

# from data_tools.dataset import DummyDataset
from data_tools.graph_tools.graph import Graph, prepare_mesh
from models.graph_attention.gan import GraphAttentionNetwork

"""
- Direct Method (CNN) for general test (CNN 256 to 64, add coord2d, Cnn to regression)
"""

# from models.graph_transformer.fast_gtn import FastGTN
import argparse
import trimesh
import sklearn


def test_trimesh():
    mesh = trimesh.load_mesh(
        "/Users/sebastian/Documents/Projects/pose_project/data/datasets/obj_000006.ply"
    )
    points, index = trimesh.sample.sample_surface_even(mesh, 100)
    triangles = mesh.triangles[index]
    pt1 = triangles[:, 0, :]
    pt2 = triangles[:, 1, :]
    pt3 = triangles[:, 2, :]
    norm = np.cross(pt3 - pt1, pt2 - pt1)
    norm = sklearn.preprocessing.normalize(norm, axis=1)
    return 0


def test_reconstruction():
    mesh = o3d.io.read_triangle_mesh(
        "/Users/sebastian/Documents/Projects/pose_project/data/datasets/obj_000006.ply"
    )
    mesh2 = o3d.io.read_triangle_mesh(
        "/Users/sebastian/Documents/Projects/pose_project/data/datasets/obj_000005.ply")
    print(np.asarray(mesh.vertices).shape)
    print(np.asarray(mesh2.vertices).shape)
    pt = mesh.sample_points_poisson_disk(number_of_points=300)
    #pt  = mesh.sample_points_uniformly(number_of_points=200)
    pt2 = mesh2.sample_points_uniformly(number_of_points=300)
    print(np.asarray(pt.points).shape) 
    print(np.asarray(pt2.points).shape)
    
    #mesh = trimesh.load_mesh(
    #    "/Users/sebastian/Documents/Projects/pose_project/data/datasets/obj_000006.ply"
    #    )
    #points, index = trimesh.sample.sample_surface_even(mesh, 100)
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(points)
    #pcd.estimate_normals()

    # hidden point removal possible for pointclouds (need to transform pose first)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pt)[0]
    mesh2 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pt2)[0]
    #mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pt, alpha=0.1)
    #o3d.visualization.draw_geometries([pt])
    #mesh = pt.compute_convex_hull()[0]
    #mesh.compute_vertex_normals()
    print(np.asarray(mesh.vertices).shape)
    print(np.asarray(mesh2.vertices).shape)
    #mesh = pt.estimate_normals()
    o3d.visualization.draw_geometries([mesh])
    return 0


import pickle

def main() -> int:

    path = "/Users/sebastian/Documents/Projects/pose_project/data/datasets/info_ellipsoid.dat"
    with open(path, "rb") as f:
        info = pickle.load(f)
    test_reconstruction()
    #test_trimesh()

    args = argparse.Namespace(
        non_local=False,
        num_channels=256,
        node_dim=256,
        num_layers=2,
    )
    # model = FastGTN(num_edge_type=1, w_in=256, num_class=6, num_nodes=100, args=args)
    #    input_ = Graph.create_random_graph(100, 256)
    #    init_garph = Graph.create_initial_graph(100, 3)
    #    batch_A = torch.from_numpy(input_.adjacency_matrix).unsqueeze(-1)
    #    batch_X = torch.from_numpy(input_.feature_matrix)
    #    batch_num_nodes = torch.tensor([100])
    #    output = model(batch_A, batch_X, batch_num_nodes)
    #
    print("halt")
    mesh = o3d.io.read_triangle_mesh(
        "/Users/sebastian/Documents/Projects/pose_project/data/datasets/obj_000006.ply"
    )
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
    mesh = prepare_mesh(
        mesh=mesh,
        #simplify_factor=10,  # 10
        num_vertices=100,
        pose=pose,
        intrinsic_matrix=cam_K,
        img_width=640,
        img_height=480,
        only_front_facing=False,
    )
    print(np.asarray(mesh.vertices).shape)
    graph = Graph.from_mesh(mesh)
    # graph.remove_unconnected_nodes()
    graph.transform_features_to_site(cam_k=cam_K, im_w=640, im_h=480)
    graph.transform_features_to_3d_coords(cam_k=cam_K, im_w=640, im_h=480)
    graph.visualize()
    random_graph = Graph.create_random_graph(100, 3)
    # graph.visualize()
    num_features = graph.num_features
    gan = GraphAttentionNetwork(
        in_features=num_features, n_hidden=256, n_classes=6, n_heads=8, dropout=0.5
    )
    # num of parameters
    print(sum(p.numel() for p in gan.parameters() if p.requires_grad))
    output = gan(
        torch.from_numpy(graph.feature_matrix),
        torch.from_numpy(graph.adjacency_matrix).unsqueeze(-1),
    )
    return 0


if __name__ == "__main__":
    main()
