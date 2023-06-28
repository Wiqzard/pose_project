from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["savefig.dpi"] = 80
mpl.rcParams["figure.dpi"] = 80

import open3d as o3d
import numpy as np
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes

path_to_model = "/home/bmw/Documents/limemod/lm/models/obj_000006.ply"
mesh = o3d.io.read_triangle_mesh(path_to_model)
vertices = torch.from_numpy(np.asarray(mesh.vertices)).float()
faces = torch.from_numpy(np.asarray(mesh.triangles)).long()

mesh = Meshes(verts=[vertices], faces=[faces])


def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 1000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title(title)
    #ax.view_init(190, 30)
    plt.show()
    plt.savefig("temp/initial_graph.png")
#sample_trg = sample_points_from_meshes(mesh, 5000)



plot_pointcloud(mesh)


print("hello")
