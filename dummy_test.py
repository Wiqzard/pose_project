
import numpy as np
import open3d as o3d
import torch

from data_tools.graph_tools.graph import Graph, prepare_mesh
from data_tools.dummy_dataset import DummyDataset

from data_tools.bop_dataset import BOPDataset
from data_tools.dataset import LMDataset

from gat_inf import GraphNet, AttentionMode, GraphUnpoolingMesh, edge_based_unpooling
from utils.flags import Mode
#sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=5)

#sphere.compute_vertex_normals()
#arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=20, cone_radius=25, cylinder_height=100, cone_height=40, resolution=10, cylinder_split=4, cone_split=1)
#arrow.compute_vertex_normals()
#arrow_graph = Graph.from_mesh(arrow)
#arrow_graph.visualize()
###visulize arrow
##o3d.visualization.draw_geometries([arrow])
#sphere = prepare_mesh(sphere)
#sphere_graph = Graph.from_mesh(sphere)

dataset = BOPDataset(
    "/home/bmw/Documents/limemod/lm",#home/bmw/Documents/limemod/lm",
    Mode.TRAIN,
    use_cache=True,
    single_object=False,
)
dummy_dataset = DummyDataset(bop_dataset=dataset)

inp = dummy_dataset[0]

arrow = Graph.from_mesh(dummy_dataset.ARROW)
arrow.set_edge_index()
arrow.visualize("temp/arrow.png")

e, x= edge_based_unpooling(torch.from_numpy(arrow.edge_index_list).T, torch.from_numpy(arrow.feature_matrix))
arrow_unpool = Graph(feature_matrix=x.numpy(), edge_index_list=e.numpy().T)
arrow_unpool.visualize("temp/arrow_unpool.png")


from torch_geometric.nn import GCNConv
from models.custom_layers import GeGLU, LearnedPositionalEmbeddings, GraphResNetBlock
gcn_conv = GCNConv(3, 16)
resnet = GraphResNetBlock(3, 16, 16)
out_i = gcn_conv(inp[1], inp[2])
out = gcn_conv(torch.from_numpy(arrow.feature_matrix).float(), torch.from_numpy(arrow.edge_index_list).long().T)
out_res = resnet(torch.from_numpy(arrow.feature_matrix).float(), torch.from_numpy(arrow.edge_index_list).long().T)

graph_net = GraphNet(
    in_channels=3, 
    out_channels=3, 
    channels=16,
    n_res_blocks=2,
    attention_levels=[1, 2],
    attention_mode=AttentionMode.GAT,
    channel_multipliers=[1, 1, 2, 2],
    unpooling_levels=[], # only in downsampling, avoid unpooling in last level
    n_heads=4,
    d_cond=1024,
)






#####################################
input_ = dummy_dataset._generate_initial_graph(0)
for i in range(1, len(dataset)):
    graph = dummy_dataset._generate_graph_for_img(i)
    feats = graph.feature_matrix
    # compute com
    com = np.mean(feats, axis=0)
    if com.max() > 2:
        print("com: ", com)
        print("index: ", i)
        print("feats: ", feats)
from pytorch3d.structures import Meshes, Pointclouds
import torch

import numpy as np

def transform_pointcloud(pointcloud, pose):
    # Ensure the pointcloud and pose are numpy arrays
    pointcloud = np.array(pointcloud)
    pose = np.array(pose)
    
    # Check that the input dimensions are correct
    assert pointcloud.shape[1] == 3
    assert pose.shape == (3, 4)

    # Homogenize the pointcloud coordinates to shape (N,4)
    homogenized_pointcloud = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1))))

    # Apply transformation: transformed_points = pose * homogenized_pointcloud
    transformed_pointcloud = np.dot(homogenized_pointcloud, pose.T)

    return transformed_pointcloud[:, :3]

randn_graph = Graph.from_mesh(dummy_dataset.ARROW)# Graph.create_random_graph(10,3)
randn_graph.visualize("t1.png")
pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
pose = np.array([[    0.85704,    0.067054,    -0.51087,       216.4],

 [   -0.33614,     -0.6787,    -0.65298,      88.047],
 [   -0.39051,     0.73135,    -0.55913,      808.46]])
transformed_pointcloud = transform_pointcloud(randn_graph.feature_matrix, pose)
randn_graph.feature_matrix = transformed_pointcloud
randn_graph.visualize("t2.png")

def graph_to_meshes(node_features, edge_index):
    """
    Transforms a graph feature matrix and edge index list consisting of multiple
    disconnected graphs into a list of PyTorch3D Meshes or Pointclouds objects,
    where each object is a connected graph.

    Args:
        node_features: tensor with node features of shape (N_nodes, Feat_dim).
        edge_index: tensor with edge indices of shape (2, Num_edges).

    Returns:
        meshes: list of PyTorch3D Meshes or Pointclouds objects.
    """
    # Determine the connected components of the graph
    components = connected_components(edge_index)

    # Initialize an empty list to store the individual Meshes or Pointclouds objects
    meshes = []

    # Loop over each connected component and create a Meshes or Pointclouds object
    for i in range(components.max() + 1):
        # Find the nodes that belong to the current component
        nodes = (components == i).nonzero().squeeze()

        # Extract the corresponding node features
        vertices = node_features[nodes]

        # Extract the corresponding edges
        edges = edge_index[:, nodes]
        edges = edges - edges.min()  # Shift edge indices so they start at 0

        # Note: PyTorch3D's Meshes data structure assumes 3D meshes
        # But if you don't have 3D data or face info, you could use Pointclouds instead
        # Here we assume we only have vertices (i.e., 3D points), and no faces
        # So we use Pointclouds to represent our graphs
        point_cloud = Pointclouds(points=vertices.unsqueeze(0))

        meshes.append(point_cloud)

    return meshes

#meshes = graph_to_meshes(torch.from_numpy(input_.feature_matrix), torch.from_numpy(input_.edge_index).T)

graph_net = GraphNet(
    in_channels=3, 
    out_channels=3, 
    channels=16,
    n_res_blocks=2,
    attention_levels=[1, 2],
    attention_mode=AttentionMode.GAT,
    channel_multipliers=[1, 1, 2, 2],
    unpooling_levels=[0, 1, 2], # only in downsampling, avoid unpooling in last level
    n_heads=4,
    d_cond=1024,
)
unpooling = GraphUnpoolingMesh()
print("graph_net params: ", sum(p.numel() for p in graph_net.parameters()))
random_graph = Graph.create_random_graph(10, 3)
random_graph.set_edge_index()
cond = torch.randn(1,49,1024)
#out2 = unpooling(torch.from_numpy(random_graph.feature_matrix), torch.from_numpy(random_graph.edge_index).T)
#edges = out2[1]
#max_index = torch.max(edges)
#assert out2[0].shape[0] == max_index + 1
#out = graph_net(torch.from_numpy(random_graph.feature_matrix), torch.from_numpy(random_graph.edge_index).T, cond=cond)


x = torch.from_numpy(input_.feature_matrix).float()
edge_index = torch.from_numpy(input_.edge_index).long().T
out = graph_net(x, edge_index, cond=cond)


final_graph = dummy_dataset._generate_graph_for_img(0)
final_graph.visualize()
inital_graph = dummy_dataset._generate_initial_graph(0)
inital_graph.visualize()
#centers = [
    #np.asarray([180 282]),
    #np.asarray([528 333]),
    #np.asarray([543 300]),
    #np.asarray([357 174]),
    #np.asarray([313 207]),
    #np.asarray([246 227]),
    #np.asarray([428 227]),
    #np.asarray([578 306]),
    #np.asarray([253 152]),
    #np.asarray([210 413]),
    #np.asarray([320 161]),
    #np.asarray([547 250]),
    #np.asarray([550 451]),
    #np.asarray([476 357]),
    #np.asarray([90 348])
    #]
print(len(dummy_dataset))









sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=5)
tetra_mesh = o3d.geometry.TriangleMesh.create_tetrahedron(radius=0.1)
#print number of vertices
print(np.asarray(sphere.vertices).shape)
print(np.asarray(tetra_mesh.vertices).shape)
tetra_mesh.compute_vertex_normals()
sphere.compute_vertex_normals()
#o3d.visualization.draw_geometries([sphere])
#o3d.visualization.draw_geometries([tetra_mesh])


sphere_graph = Graph.from_mesh(sphere)
sphere_graph.set_edge_index()# set_adjacency_list()

tetra_graph = Graph.from_mesh(tetra_mesh)
tetra_graph.set_edge_index()# set_adjacency_list()

import torch

def edge_based_unpool(edge_list, X):
    # The number of original vertices and edges
    num_vertices = X.size(0)
    num_edges = edge_list.size(1)

    # Calculate the new features for the added vertices
    new_features = (X[edge_list[0]] + X[edge_list[1]]) / 2.0

    # Concatenate the original features with the new ones
    X_unpooled = torch.cat([X, new_features], dim=0)

    # Create new edges for the added vertices
    new_edge_list = torch.cat([edge_list, torch.arange(num_vertices, num_vertices + num_edges).unsqueeze(0).repeat(2,1)], dim=1)

    # Create the connections between the new vertices
    new_triangle_edges = torch.stack([edge_list[0], edge_list[1], torch.arange(num_vertices, num_vertices + num_edges)], dim=1)
    new_triangle_edges = new_triangle_edges[:, [0,2,1,2]].view(-1, 2).t()

    # Concatenate the original edges with the new ones
    edge_list_unpooled = torch.cat([new_edge_list, new_triangle_edges], dim=1)

    return X_unpooled, edge_list_unpooled

def edge_based_unpool2(edge_index, X):
    # The number of original vertices and edges
    num_vertices = X.shape[0]
    num_edges = edge_index.shape[0]

    # Calculate the new features for the added vertices
    new_features = (X[edge_index[:,0]] + X[edge_index[:,1]]) / 2.0

    # Concatenate the original features with the new ones
    X_unpooled = np.concatenate([X, new_features], axis=0)

    # Create new edges for the added vertices
    new_edge_list = np.concatenate([edge_index, np.repeat(np.arange(num_vertices, num_vertices + num_edges).reshape(-1,1), 2, axis=1)], axis=0)

    # Create the connections between the new vertices
    new_triangle_edges = np.stack([edge_index[:,0], edge_index[:,1], np.arange(num_vertices, num_vertices + num_edges)], axis=1)
    new_triangle_edges = new_triangle_edges[:, [0,2,1,2]].reshape(-1, 2)

    # Concatenate the original edges with the new ones
    edge_list_unpooled = np.concatenate([new_edge_list, new_triangle_edges], axis=0)

    return X_unpooled, edge_list_unpooled


import numpy as np





def edge_list_to_adjacency_matrix(edge_list):
    num_nodes = max(max(edge_list[:, 0]), max(edge_list[:, 1])) + 1
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for edge in edge_list:
        node1, node2 = edge[0], edge[1]
        adj_matrix[node1, node2] = 1
        adj_matrix[node2, node1] = 1  # If your graph is undirected, uncomment this line
    return adj_matrix#
# 4 -> 16 -> 64 -> 256
# centers = numpy array of shape (N, 2) 
centers = np.array([[0,0], [0.1,0.1], [0.5,0.5], [1,1]])
init_graph = Graph.create_initial_graph_2(centers)
init_graph.set_edge_index()
init_graph_unpooled2 = edge_based_unpooling(init_graph.edge_index, init_graph.feature_matrix)
#init_graph_unpooled2 = edge_based_unpool(init_graph_unpooled[1], init_graph_unpooled[0])

init_graph2 = Graph(adjacency_matrix=edge_index_to_adjacency_matrix(init_graph_unpooled2[0]), feature_matrix=init_graph_unpooled2[1])

mesh = o3d.geometry.TriangleMesh.create_mobius(length_split=30, width_split=5) 
graph = Graph.from_mesh(mesh)

graph.set_edge_index()
graph_unpooled = edge_based_unpooling(graph.edge_index, graph.feature_matrix)
#graph_unpooled2 = edge_based_unpooling(graph_unpooled[0], graph_unpooled[1])
graph_unpooled = Graph(adjacency_matrix=edge_index_to_adjacency_matrix(graph_unpooled[0]), feature_matrix=graph_unpooled[1])
graph_unpooled.visualize()

init_graph2.visualize()
print("done")