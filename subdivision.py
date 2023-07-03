import torch
import torch.nn as nn

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

class SubdivideMeshes(nn.Module):
    def __init__(self) -> None:
        super(SubdivideMeshes, self).__init__()

    def subdivide_faces(self, num_verts, faces, face_edge_rep):
        """
        Args:
            verts: a (N, 3) Tensor of vertices.
            faces: a (F, 3) Tensor of faces.

        Returns:
            subdivided_faces: (4*F, 3) shape LongTensor of original and new faces.
        """
        with torch.no_grad():
            faces_to_edges = face_edge_rep + num_verts

            f0 = torch.stack(
                [
                    faces[:, 0],
                    faces_to_edges[:, 2],
                    faces_to_edges[:, 1],
                ],
                dim=1,
            )
            f1 = torch.stack(
                [
                    faces[:, 1],
                    faces_to_edges[:, 0],
                    faces_to_edges[:, 2],
                ],
                dim=1,
            )
            f2 = torch.stack(
                [
                    faces[:, 2],
                    faces_to_edges[:, 1],
                    faces_to_edges[:, 0],
                ],
                dim=1,
            )
            f3 = faces_to_edges
            subdivided_faces = torch.cat(
                [f0, f1, f2, f3], dim=0
            )

            return subdivided_faces

#    def faces_to_edges(self, faces, edges):
#        V = edges.max() + 1
#        edges_hash = V * edges[ :, 0] + edges[ :, 1]
#        u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
#        #sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
#        #unique_mask = torch.ones(edges_hash.shape[0], dtype=torch.bool)
#        #unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
#        #unique_idx = sort_idx[unique_mask]
#        ##edges = torch.stack([u // V, u % V], dim=1)
#        #edges_packed_to_mesh_idx = edges[unique_idx]
#        F = faces.shape[0]
#        faces_edge_rep = inverse_idxs.reshape(3, F).t()
#        return faces_edge_rep 
#
    def faces_to_edges(self, num_verts, faces):
        edges1 = faces[:, [0, 1]]
        edges2 = faces[:, [1, 2]]
        edges3 = faces[:, [2, 0]]
        edges = torch.cat([edges1, edges2, edges3], dim=0)
        edges, _ = edges.sort(dim=1)
        edges_hash = num_verts * edges[:, 0] + edges[:, 1]
        u, inverse_idxs  = torch.unique(edges_hash, return_inverse=True)
        F = faces.shape[0]
        faces_edge_rep = inverse_idxs.reshape(3, F).t()
        edges = torch.stack([u // num_verts, u % num_verts], dim=1)
        return edges, faces_edge_rep

    def forward(self, verts, edges, faces, feats=None):
        """
        Subdivide a batch of meshes by adding a new vertex on each edge, and
        dividing each face into four new faces. 

        Args:
            verts: (B, N, 3) Tensor representing vertices.
            edges: (B * E, 2) Tensor representing edges.
            faces: (B * F, 3) Tensor representing faces.
            feats: (optional) Per-vertex features to be subdivided along with the verts.

        Returns:
            2-element tuple containing

            - **new_verts**: Tensor of vertices in the subdivided meshes.
            - **new_faces**: Tensor of faces in the subdivided meshes.
            - **new_feats**: (optional) Tensor of subdivided features. Only returned if feats is not None.
        """
        bs, num_vertices = verts.shape[:2]
        edges_single = edges[(edges[:, 0] < num_vertices) & (edges[:, 1] < num_vertices)]
        #faces_single = faces[(faces[:, 0] < num_vertices) & (faces[:, 1] < num_vertices) & (faces[:, 2] < num_vertices)]

        #_, faces_edges = self.faces_to_edges(num_vertices, faces_single)
        _, faces_edges = self.faces_to_edges(bs * num_vertices, faces)


        new_faces = self.subdivide_faces(num_vertices, faces, faces_edges)
        new_edges, _ = self.faces_to_edges(bs * num_vertices, new_faces)
        #new_faces = new_faces.view(1, -1, 3).expand(bs, -1, -1)
        # Add one new vertex at the midpoint of each edge by taking the average
        # of the vertices that form each edge.
        new_verts = verts[:, edges_single].mean(dim=2)
        new_verts = torch.cat([verts, new_verts], dim=1)  # (sum(V_n)+sum(E_n), 3)

        new_feats = None
        if feats is not None:
            if feats.dim() == 2:
                feats = feats.view(verts.size(0), verts.size(1), feats.size(1))
            if feats.dim() != 3:
                raise ValueError("features need to be of shape (N, V, D) or (N*V, D)")

            new_feats = torch.cat([feats, feats[:, edges].mean(dim=2)], dim=1)

        if feats is None:
            return new_verts, new_faces ,new_edges.T
        else:
            return new_verts, new_faces, new_edges.T, new_feats

#class SubdivideMeshes(nn.Module):
    #def __init__(self) -> None:
        #super(SubdivideMeshes, self).__init__()

    #def subdivide_faces(self, verts, faces, face_edge_rep):
        #"""
        #Args:
            #verts: a (N, 3) Tensor of vertices.
            #faces: a (F, 3) Tensor of faces.

        #Returns:
            #subdivided_faces: (4*F, 3) shape LongTensor of original and new faces.
        #"""
        #with torch.no_grad():
            #faces_to_edges = face_edge_rep + verts.shape[0]

            #f0 = torch.stack(
                #[
                    #faces[:, 0],
                    #faces_to_edges[:, 2],
                    #faces_to_edges[:, 1],
                #],
                #dim=1,
            #)
            #f1 = torch.stack(
                #[
                    #faces[:, 1],
                    #faces_to_edges[:, 0],
                    #faces_to_edges[:, 2],
                #],
                #dim=1,
            #)
            #f2 = torch.stack(
                #[
                    #faces[:, 2],
                    #faces_to_edges[:, 1],
                    #faces_to_edges[:, 0],
                #],
                #dim=1,
            #)
            #f3 = faces_to_edges
            #subdivided_faces = torch.cat(
                #[f0, f1, f2, f3], dim=0
            #)

            #return subdivided_faces

##    def faces_to_edges(self, faces, edges):
##        V = edges.max() + 1
##        edges_hash = V * edges[ :, 0] + edges[ :, 1]
##        u, inverse_idxs = torch.unique(edges_hash, return_inverse=True)
##        #sorted_hash, sort_idx = torch.sort(edges_hash, dim=0)
##        #unique_mask = torch.ones(edges_hash.shape[0], dtype=torch.bool)
##        #unique_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
##        #unique_idx = sort_idx[unique_mask]
##        ##edges = torch.stack([u // V, u % V], dim=1)
##        #edges_packed_to_mesh_idx = edges[unique_idx]
##        F = faces.shape[0]
##        faces_edge_rep = inverse_idxs.reshape(3, F).t()
##        return faces_edge_rep 
##
    #def faces_to_edges(self, num_verts, faces):
        #edges1 = faces[:, [0, 1]]
        #edges2 = faces[:, [1, 2]]
        #edges3 = faces[:, [2, 0]]
        #edges = torch.cat([edges1, edges2, edges3], dim=0)
        #edges, _ = edges.sort(dim=1)
        #edges_hash = num_verts * edges[:, 0] + edges[:, 1]
        #u, inverse_idxs  = torch.unique(edges_hash, return_inverse=True)
        #F = faces.shape[0]
        #faces_edge_rep = inverse_idxs.reshape(3, F).t()
        #edges = torch.stack([u // num_verts, u % num_verts], dim=1)
        #return edges, faces_edge_rep

    #def forward(self, verts, edges, faces, feats=None):
        #"""
        #Subdivide a batch of meshes by adding a new vertex on each edge, and
        #dividing each face into four new faces. 

        #Args:
            #verts: (N, 3) Tensor representing vertices.
            #edges: (E, 2) Tensor representing edges.
            #faces: (F, 3) Tensor representing faces.
            #feats: (optional) Per-vertex features to be subdivided along with the verts.

        #Returns:
            #2-element tuple containing

            #- **new_verts**: Tensor of vertices in the subdivided meshes.
            #- **new_faces**: Tensor of faces in the subdivided meshes.
            #- **new_feats**: (optional) Tensor of subdivided features. Only returned if feats is not None.
        #"""
        #num_vertices = verts.shape[0]
        #edges, faces_edges = self.faces_to_edges(num_vertices, faces)
        #new_faces = self.subdivide_faces(verts,faces, faces_edges)
        #new_verts = torch.cat([verts, verts[edges].mean(dim=1)], dim=0)

        #new_feats = None
        #if feats is not None:
            #if feats.dim() == 2:
                #feats = feats.view(verts.size(0), verts.size(1), feats.size(1))
            #if feats.dim() != 3:
                #raise ValueError("features need to be of shape (N, V, D) or (N*V, D)")

            #new_feats = torch.cat([feats, feats[:, edges].mean(dim=2)], dim=1)

        #if feats is None:
            #return new_verts, new_faces ,edges.T
        #else:
            #return new_verts, new_faces,edges.T, new_feats
