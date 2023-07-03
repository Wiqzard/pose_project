

from pathlib import Path
import numpy as np
from typing import Union

import trimesh
# Get Initial Graph and Ground Truth graph
from data_tools.bop_dataset import BOPDataset, Mode
from data_tools.dummy_dataset import edge_based_unpooling
from data_tools.graph_tools.graph import Graph
import open3d as o3d
import torch
import timm
from gat_inf import SpatialTransformer, GraphNet, GraphNetv2, AttentionMode

points = np.load("/home/bmw/Documents/Sebastian/pose_project/predx.npy")#[0]
points_gt = np.load("/home/bmw/Documents/Sebastian/pose_project/targetx.npy")#[0]

#points = np.load("/home/bmw/Documents/Sebastian/pose_project/init.npy")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def visualize_and_save(pointcloud, filename):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Point cloud is of shape (N,3gt)
    x = pointcloud[:,0]
    y = pointcloud[:,1]
    z = pointcloud[:,2]

    ax.scatter(x, y, z, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(filename)
    plt.show()
visualize_and_save(points, "predx.png")
visualize_and_save(points_gt, "gtx.png")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
vis = o3d.visualization.Visualizer()
pred_edges = np.load("/home/bmw/Documents/Sebastian/pose_project/pred_edges.npy")#[0]
gt_points = np.load("/home/bmw/Documents/Sebastian/pose_project/gt.npy")#[0]
gt_normals = torch.from_numpy(gt_points[1]).squeeze()
gt_points = torch.from_numpy(gt_points[0]).squeeze()

pred_points = torch.from_numpy(points[0]).squeeze()




#dataset = BOPDataset(
#    "/home/bmw/Documents/limemod/lm",#home/bmw/Documents/limemod/lm",
#    Mode.TRAIN,
#    use_cache=True,
#    single_object=False,
#    num_points=5082,
#)
#from data_tools.dataset import DatasetLM
#
#dataset = DatasetLM(dataset)
#a = dataset[0]
#
#a = dataset.get_graph_gt_path(378)
a = np.load('/home/bmw/Documents/limemod/lm/train_pbr/000037/graphs/init/init_graph_000000.npy')
features = a["initial_features"][0]
edge_index = a["initial_edges"][0]
spat = SpatialTransformer(channels=3, n_heads=3, n_layers=3,d_cond=768)
model = GraphNet
cond= torch.rand(1,49,768)
features = torch.from_numpy(features)
edge_index = torch.from_numpy(edge_index)
a = spat(features ,edge_index.T,cond)
backbone = timm.create_model(
    model_name="maxxvitv2_rmlp_base_rw_224.sw_in12k_ft_in1k",
    pretrained=True,
)
graph_net = GraphNet(
    backbone=backbone,
    in_channels=3, 
    out_channels=3, 
    channels=16,
    n_res_blocks=2,
    attention_levels=[1, 2],
    attention_mode=AttentionMode.GAT,
    channel_multipliers=[1, 1, 2, 2],
    unpooling_levels=[], # only in downsampling, avoid unpooling in last level
    n_heads=4,
    d_cond=768,
)
graph_netv2 = GraphNetv2(
    backbone=backbone,
    in_channels=3,
    out_channels=3,
    d_model=768,
    n_res_blocks=2,
    attention_levels=[1, 2],
    unpooling_levels=[2,5, 6],
    channel_multipliers=[1,1, 2,2, 4 ,8, 16, 32],
    n_heads=4,
    channels=16,
    d_cond=768,
)
b = graph_netv2(features, edge_index.T, cond=cond)
# print graph net parameters
print(
    "number of parameters: ", sum(p.numel() for p in graph_netv2.parameters() if p.requires_grad)
)
b= graph_net(features, edge_index.T, cond=cond)
# print graph net parameters
print(
    "number of parameters: ", sum(p.numel() for p in graph_net.parameters() if p.requires_grad)
) 



a = np.load('/home/bmw/Documents/limemod/lm/train_pbr/000037/graphs/init/init_graph_000000.npy')
features = a["initial_features"][0]
edge_index = a["initial_edges"][0]

sphere = trimesh.creation.icosphere(radius=0.05, subdivisions=1)
edges = sphere.edges
vertices = np.asarray(sphere.vertices, dtype=np.float32)
pool_edge_index, pool_features = edge_based_unpooling(edges, vertices)
pool_edge_index, pool_features = edge_based_unpooling(pool_edge_index, pool_features)
pool_edge_index, pool_features = edge_based_unpooling(pool_edge_index, pool_features)
#graph = Graph(edge_index_list=pool_edge_index, feature_matrix=pool_features)
#graph.visualize()   

pool_edge_index, pool_features = edge_based_unpooling(edge_index, features)

dataset.generate_initial_graphs(img_height=480, img_width=640)
dataset.generate_gt_graphs(num_points=5082, img_height=480, img_width=640, site=True)

# 

# Transform Vertices to Camera Coordinate System
# Vertices, Vertex Normals are ground truth
# before caching, load models, calculate vertex normals, sample points, save vertices and vertex normals to cache


path_to_models = Path("/home/bmw/Documents/limemod/lm/models")
assert path_to_models.exists()
model_paths = list(path_to_models.glob("*.ply"))
meshes  = [trimesh.load_mesh(str(path)) for path in model_paths]
vertices = [np.array(mesh.vertices) for mesh in meshes]   
normals = [np.array(mesh.vertex_normals) for mesh in meshes]
num_samples = 100
a = trimesh.sample.sample_surface_even(meshes[0], num_samples)
import numpy as np
import trimesh

import open3d as o3d



def func1(model_path: Union[str, Path], pose: np.ndarray, num_samples: int = 1000):
    mesh = o3d.io.read_triangle_mesh(str(model_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    point_cloud = mesh.sample_points_poisson_disk(1000)
    #mesh.normalize_normals() 
    vertices = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    # transform pointcloud to camera coordinate system 
    pose[:3, 3] = pose[:3, 3] / 1000
    homogenized_pointcloud = np.hstack((vertices/1000, np.ones((vertices.shape[0], 1))))
    transformed_pointcloud = np.dot(homogenized_pointcloud, pose.T)#[:,:3]
    transformed_normals = np.dot(normals, pose[:3, :3].T)
    
    # create a point cloud
    pcd = o3d.geometry.PointCloud()

    # assign the vertices and normals
    pcd.points = o3d.utility.Vector3dVector(transformed_pointcloud)
    pcd.normals = o3d.utility.Vector3dVector(transformed_normals)

    # visualize the point cloud
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    

    print(mesh.has_vertex_normals())

pose = np.array([[    0.85704,    0.067054,    -0.51087,       216.4],
                [   -0.33614,     -0.6787,    -0.65298,      88.047],
                [   -0.39051,     0.73135,    -0.55913,      808.46]])

func1(model_paths[8], pose, 1000)

import open3d as o3d

from scipy.spatial import cKDTree

print("halt")
 

