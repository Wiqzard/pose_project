import numpy as np
import open3d as o3d

# from data_tools.dataset import DummyDataset
from data_tools.keypoints import Graph, prepare_mesh


"""
- Direct Method (CNN) for general test (CNN 256 to 64, add coord2d, Cnn to regression)
"""


def main() -> int:
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
        simplify_factor=10,
        pose=pose,
        intrinsic_matrix=cam_K,
        img_width=640,
        img_height=480,
    )
    graph = Graph.from_mesh(mesh)
    graph.remove_unconnected_nodes()
    # graph.visualize()
    return 0


if __name__ == "__main__":
    main()
