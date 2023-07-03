from data_tools.bop_dataset import BOPDataset
from data_tools.dataset import LMDataset, DatasetLM
from utils.flags import Mode

import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from data_tools.graph_tools.graph import Graph

from pytorch3d.io import load_obj
from pytorch3d.utils.ico_sphere import ico_sphere
from pytorch3d.structures import Meshes
#from pytorch3d.ops import SubdivideMeshes
from subdivision import SubdivideMeshes


sphere = ico_sphere(0)
sphere_verts = sphere.verts_packed()
sphere_faces = sphere.faces_packed()
sphere_edges = sphere.edges_packed()

# 5 meshes as batch
sphere_verts = sphere_verts.unsqueeze(0).repeat(5, 1, 1)
sphere_faces = sphere_faces.unsqueeze(0).repeat(5, 1, 1)
sphere_edges = sphere_edges.unsqueeze(0).repeat(5, 1, 1)
#
mesh = Meshes(verts=sphere_verts, faces=sphere_faces)
verts = mesh.verts_packed()
verts_pad = mesh.verts_padded()
#divider = SubdivideMeshes(mesh)
divider = SubdivideMeshes()
mesh = divider(mesh.verts_padded(), mesh.edges_packed(), mesh.faces_packed())
verts2 = mesh.verts_packed()
verts_pad2 = mesh.verts_padded()

print(sphere_verts)
print(sphere_faces)
print(sphere_edges)
print("s")

def faces_to_edges_torch(faces):
    edges1 = faces[:, [0, 1]]
    edges2 = faces[:, [1, 2]]
    edges3 = faces[:, [2, 0]]
    edges = torch.cat([edges1, edges2, edges3], dim=0)
    edges, _ = edges.sort(dim=1)
    V = edges.max() + 1
    edges_hash = V * edges[:, 0] + edges[:, 1]
    u, _ = torch.unique(edges_hash, return_inverse=True)
    edges = torch.stack([u // V, u % V], dim=1)
    return edges


def compute_vertex_normals(verts, faces):
    """
    Compute vertex normals for a mesh (3D).
    
    Args:
        verts: FloatTensor of shape (V, 3) giving vertex positions
        faces: LongTensor of shape (F, 3) giving faces (must be triangles)
    
    Returns:
        verts_normals: FloatTensor of shape (V, 3) giving vertex normals 
    """
    verts_normals = torch.zeros_like(verts)
    vertices_faces = verts[faces]
    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )
    verts_normals = verts_normals.index_add(
        0, faces[:, 0], faces_normals
    )
    verts_normals = verts_normals.index_add(
        0, faces[:, 1], faces_normals
    )
    verts_normals = verts_normals.index_add(
        0, faces[:, 2], faces_normals
    )

    verts_normals_norm = torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )
    return verts_normals_norm

#normals = faces_to_normals(_ico_verts0, _ico_faces0)
# edges = faces_to_edges(_ico_faces0)
edges2 = faces_to_edges_torch(torch.tensor(_ico_faces0))
# check if edges edges2 are the same

assert (edges2 == torch.tensor(_ico_edges0)).all()

verts = torch.tensor(_ico_verts0, dtype=torch.float32)
faces = torch.tensor(_ico_faces0, dtype=torch.int64)
edges = torch.tensor(_ico_edges0, dtype=torch.int64)
from subdivision import SubdivideMeshes
normals = compute_vertex_normals(verts, faces)
subdiv = SubdivideMeshes()
forw = subdiv(verts, edges, faces)



edge_index = edges
num_nodes = verts.shape[0]
num_edges = edge_index.shape[0]
new_node_coords = torch.mean(verts[edge_index], dim=1)
new_feature_matrix = torch.cat([verts, new_node_coords], dim=0)
verts2 = torch.unique(new_feature_matrix, dim=0)

graph = Graph(feature_matrix=verts2.numpy(), edge_index_list=sphere_edges.numpy())
graph.visualize("sphere2.png")


from gat_inf import edge_based_unpooling

e_i, feat = edge_based_unpooling(edges.T, verts)
print("halt")
import torch
from collections import defaultdict


def find_faces(N, adjacency_list):
    # Create a dictionary mapping each edge to its adjacent vertices
    edge_to_vertices = defaultdict(list)

    for i in range(N):
        for j in adjacency_list[i]:
            # Since the mesh is undirected, sort the vertices
            # so we treat the edge (i, j) as the same as (j, i)
            edge = tuple(sorted((i, j)))
            edge_to_vertices[edge].append(i)
            edge_to_vertices[edge].append(j)

    # For each vertex, look up its adjacent vertices, and find groups
    # of 3 vertices that share an edge
    faces = []

    for vertex, edges in edge_to_vertices.items():
        # Create a set of adjacent vertices for easy lookup
        adjacent_vertices = set(edges)

        # For each pair of adjacent vertices, check if they share an edge
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                if edges[j] in adjacent_vertices:
                    # If they do, we've found a face
                    face = [vertex, edges[i], edges[j]]
                    face.sort()  # Sort the face vertices for consistency
                    faces.append(face)

    # Convert faces to a PyTorch tensor and return it
    faces_tensor = torch.tensor(faces, dtype=torch.long)
    return faces_tensor


adj_list = [[] for _ in range(verts.shape[0])]
for i in range(edges.shape[1]):
    edge = edges[i]
    adj_list[edge[0]].append(edge[1])
    adj_list[edge[1]].append(edge[0])
faces = find_faces(verts.shape[0], adj_list)


def find_faces(feature_matrix, edge_index):
    N = feature_matrix.shape[0]
    L = edge_index.shape[0]

    # Create an adjacency list
    adj_list = [[] for _ in range(N)]
    for i in range(L):
        edge = edge_index[i]
        adj_list[edge[0]].append(edge[1])
        adj_list[edge[1]].append(edge[0])

    # Create all possible faces and sort the vertices
    possible_faces = []
    for node, neighbors in enumerate(adj_list):
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                face = sorted([node, neighbors[i], neighbors[j]])
                possible_faces.append(face)

    # Remove duplicate faces
    faces = torch.unique(torch.tensor(possible_faces), dim=0)

    return faces


faces_calc = find_faces(verts, edges)


def find_triangles(e, edge_index):
    # Get number of nodes
    num_nodes = edge_index.max().item() + 1

    # Create an adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    adj_matrix[edge_index[:, 0], edge_index[:, 1]] = 1
    adj_matrix[edge_index[:, 1], edge_index[:, 0]] = 1  # Assuming undirected graph

    # Initialize empty list to store triangles
    triangles = []

    # Iterate over all edges
    for i in range(edge_index.shape[0]):
        # Get endpoints of the edge
        node1, node2 = edge_index[i]

        # Get neighbors of node1 and node2
        neighbors_node1 = torch.where(adj_matrix[node1, :])[0]
        neighbors_node2 = torch.where(adj_matrix[node2, :])[0]

        # Find common neighbors
        common_neighbors = torch.intersect(neighbors_node1, neighbors_node2)

        # For each common neighbor, form a triangle
        for neighbor in common_neighbors:
            triangles.append([node1.item(), node2.item(), neighbor.item()])

    return triangles


faces_calc = find_triangles(verts, edges)
# check if faces are the same
assert torch.all(faces == faces_calc), "faces are not the same"
print("faces are the same")


def calculate_vertex_normals(vertices, edges):
    edges = edges.T
    # Determine the triangles for each edge
    triangles = torch.stack(
        (edges[:, 0], edges[:, 1], torch.roll(edges[:, 1], shifts=-1)), dim=1
    )

    # Calculate vectors between vertices
    vec1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    vec2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]

    print(torch.tensor([0, 0, 0]) in vec1)
    print(torch.tensor([0, 0, 0]) in vec2)
    # Calculate normals for each triangle by cross product
    face_normals = torch.cross(vec1, vec2)

    vertex_normals = torch.zeros_like(vertices)
    for i, triangle in enumerate(triangles):
        vertex_normals[triangle] += face_normals[i]
    # check if 0 normals are present
    # assert torch.any(torch.all(torch.zeros(3) == vertex_normals), dim=1), "0 in vertex normals"
    vertex_normals /= torch.norm(vertex_normals, dim=1, keepdim=True)
    assert not torch.any(torch.isnan(vertex_normals)), "NaNs in vertex normals"

    return vertex_normals


from pytorch3d.loss import chamfer_distance

dataset = BOPDataset(
    "/home/bmw/Documents/limemod/lm",  # home/bmw/Documents/limemod/lm",
    Mode.TRAIN,
    use_cache=True,
    single_object=False,
)
dummy_dataset = DatasetLM(bop_dataset=dataset)
inp = dummy_dataset[0]
feat, edge = inp[1], inp[2]
# check if duplicate feats are present


gt_feat, gt_normals = inp[3], inp[4]
normals = calculate_vertex_normals(feat, edge)

a = 1


def visualize_normals(feat, normals, name="normals.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.quiver(
        feat[:, 0],
        feat[:, 1],
        feat[:, 2],
        normals[:, 0],
        normals[:, 1],
        normals[:, 2],
        length=0.1,
        normalize=True,
    )

    ax.scatter(feat[:, 0], feat[:, 1], feat[:, 2], c="r", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig(name)


visualize_normals(gt_feat.numpy(), gt_normals.numpy(), name="gt_normals.png")
visualize_normals(feat.numpy(), normals.numpy())

print(inp[0].shape)
