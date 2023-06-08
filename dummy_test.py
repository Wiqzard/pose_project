
import numpy as np
import open3d as o3d

from data_tools.graph_tools.graph import Graph, prepare_mesh
from data_tools.dummy_dataset import DummyDataset

from data_tools.bop_dataset import BOPDataset, Flag
from data_tools.dataset import LMDataset

dataset = BOPDataset(
    "/Users/sebastian/Documents/Projects/pose_project/data/datasets/lm",
    Flag.TRAIN,
    use_cache=True,
    single_object=False,
)
dummy_dataset = DummyDataset(bop_dataset=dataset)
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