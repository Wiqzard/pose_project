from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import math
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, quat2mat


def rot6d_to_mat_batch(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix.
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks. CVPR 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # poses
    x_raw = d6[..., 0:3]  # bx3
    y_raw = d6[..., 3:6]  # bx3

    x = F.normalize(x_raw, p=2, dim=-1)  # bx3
    z = torch.cross(x, y_raw, dim=-1)  # bx3
    z = F.normalize(z, p=2, dim=-1)  # bx3
    y = torch.cross(z, x, dim=-1)  # bx3

    # (*,3)x3 --> (*,3,3)
    return torch.stack((x, y, z), dim=-1)  # (b,3,3)


def quat2mat_torch(quat, eps=0.0):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    norm_quat = quat.norm(p=2, dim=1, keepdim=True)
    norm_quat = quat / (norm_quat + eps)
    qw, qx, qy, qz = (
        norm_quat[:, 0],
        norm_quat[:, 1],
        norm_quat[:, 2],
        norm_quat[:, 3],
    )
    B = quat.size(0)

    s = 2.0  # * Nq = qw*qw + qx*qx + qy*qy + qz*qz
    X = qx * s
    Y = qy * s
    Z = qz * s
    wX = qw * X
    wY = qw * Y
    wZ = qw * Z
    xX = qx * X
    xY = qx * Y
    xZ = qx * Z
    yY = qy * Y
    yZ = qy * Z
    zZ = qz * Z
    rotMat = torch.stack(
        [
            1.0 - (yY + zZ),
            xY - wZ,
            xZ + wY,
            xY + wZ,
            1.0 - (xX + zZ),
            yZ - wX,
            xZ - wY,
            yZ + wX,
            1.0 - (xX + yY),
        ],
        dim=1,
    ).reshape(B, 3, 3)
    return rotMat


def get_rot_mat(rot, rot_type):
    if rot_type in ["ego_quat", "allo_quat"]:
        rot_m = quat2mat_torch(rot)
    elif rot_type in ["ego_rot6d", "allo_rot6d"]:
        rot_m = rot6d_to_mat_batch(rot)
    else:
        raise ValueError(f"Wrong pred_rot type: {rot_type}")
    return rot_m


def quat2mat_torch(quat, eps=0.0):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    norm_quat = quat.norm(p=2, dim=1, keepdim=True)
    norm_quat = quat / (norm_quat + eps)
    qw, qx, qy, qz = (
        norm_quat[:, 0],
        norm_quat[:, 1],
        norm_quat[:, 2],
        norm_quat[:, 3],
    )
    B = quat.size(0)

    s = 2.0  # * Nq = qw*qw + qx*qx + qy*qy + qz*qz
    X = qx * s
    Y = qy * s
    Z = qz * s
    wX = qw * X
    wY = qw * Y
    wZ = qw * Z
    xX = qx * X
    xY = qx * Y
    xZ = qx * Z
    yY = qy * Y
    yZ = qy * Z
    zZ = qz * Z
    rotMat = torch.stack(
        [
            1.0 - (yY + zZ),
            xY - wZ,
            xZ + wY,
            xY + wZ,
            1.0 - (xX + zZ),
            yZ - wX,
            xZ - wY,
            yZ + wX,
            1.0 - (xX + yY),
        ],
        dim=1,
    ).reshape(B, 3, 3)
    return rotMat


def rot6d_to_mat_batch(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix.
    Args:
        d6: 6D rotation representation, of size (*, 6)
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks. CVPR 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    # poses
    x_raw = d6[..., 0:3]  # bx3
    y_raw = d6[..., 3:6]  # bx3

    x = F.normalize(x_raw, p=2, dim=-1)  # bx3
    z = torch.cross(x, y_raw, dim=-1)  # bx3
    z = F.normalize(z, p=2, dim=-1)  # bx3
    y = torch.cross(z, x, dim=-1)  # bx3

    # (*,3)x3 --> (*,3,3)
    return torch.stack((x, y, z), dim=-1)  # (b,3,3)


def quat_trans_to_pose_m(quat, trans):
    se3_mx = np.zeros((3, 4))
    # quat = quat / LA.norm(quat)
    R = quat2mat(quat)  # normalize internally
    se3_mx[:, :3] = R
    se3_mx[:, 3] = trans
    return se3_mx


def allocentric_to_egocentric(
    allo_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)
):
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
    if src_type == "mat":
        trans = allo_pose[:3, 3]
    elif src_type == "quat":
        trans = allo_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount

    if angle > 0:
        if dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
            if src_type == "mat":
                ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
            elif src_type == "quat":
                ego_pose[:3, :3] = np.dot(rot_mat, quat2mat(allo_pose[:4]))
        elif dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), angle)
            if src_type == "quat":
                ego_pose[:4] = qmult(rot_q, allo_pose[:4])
            elif src_type == "mat":
                ego_pose[:4] = qmult(rot_q, mat2quat(allo_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:  # allo to ego
        if src_type == "mat" and dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[:4] = mat2quat(allo_pose[:3, :3])
            ego_pose[4:7] = allo_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, :3] = quat2mat(allo_pose[:4])
            ego_pose[:3, 3] = allo_pose[4:7]
        else:
            ego_pose = allo_pose.copy()
    return ego_pose


def egocentric_to_allocentric(
    ego_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)
):
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = ego_pose[:3, 3]
    elif src_type == "quat":
        trans = ego_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount
    if angle > 0:
        if dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=-angle)
            if src_type == "mat":
                allo_pose[:3, :3] = np.dot(rot_mat, ego_pose[:3, :3])
            elif src_type == "quat":
                allo_pose[:3, :3] = np.dot(rot_mat, quat2mat(ego_pose[:4]))
        elif dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), -angle)
            if src_type == "quat":
                allo_pose[:4] = qmult(rot_q, ego_pose[:4])
            elif src_type == "mat":
                allo_pose[:4] = qmult(rot_q, mat2quat(ego_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:
        if src_type == "mat" and dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[:4] = mat2quat(ego_pose[:3, :3])
            allo_pose[4:7] = ego_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, :3] = quat2mat(ego_pose[:4])
            allo_pose[:3, 3] = ego_pose[4:7]
        else:
            allo_pose = ego_pose.copy()
    return allo_pose


def quatmul_torch(q1, q2):
    """Computes the multiplication of two quaternions.
    Note, output dims: NxMx4 with N being the batchsize and N the number
    of quaternions or 3D points to be transformed.
    """
    # RoI dimension. Unsqueeze if not fitting.
    a = q1.unsqueeze(0) if q1.dim() == 1 else q1
    b = q2.unsqueeze(0) if q2.dim() == 1 else q2

    # Corner dimension. Unsequeeze if not fitting.
    a = a.unsqueeze(1) if a.dim() == 2 else a
    b = b.unsqueeze(1) if b.dim() == 2 else b

    # Quaternion product
    x = (
        a[:, :, 1] * b[:, :, 0]
        + a[:, :, 2] * b[:, :, 3]
        - a[:, :, 3] * b[:, :, 2]
        + a[:, :, 0] * b[:, :, 1]
    )
    y = (
        -a[:, :, 1] * b[:, :, 3]
        + a[:, :, 2] * b[:, :, 0]
        + a[:, :, 3] * b[:, :, 1]
        + a[:, :, 0] * b[:, :, 2]
    )
    z = (
        a[:, :, 1] * b[:, :, 2]
        - a[:, :, 2] * b[:, :, 1]
        + a[:, :, 3] * b[:, :, 0]
        + a[:, :, 0] * b[:, :, 3]
    )
    w = (
        -a[:, :, 1] * b[:, :, 1]
        - a[:, :, 2] * b[:, :, 2]
        - a[:, :, 3] * b[:, :, 3]
        + a[:, :, 0] * b[:, :, 0]
    )

    return torch.stack((w, x, y, z), dim=2)


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def allocentric_to_egocentric_torch(translation, q_allo, eps=1e-4):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.
    Since objects further away from the optical center undergo skewing, we try to visually correct by
    rotating back the amount between optical center ray and object centroid ray.
    Another way to solve that might be translational variance (https://arxiv.org/abs/1807.03247)
    Args:
        translation: Nx3
        q_allo: Nx4
    """

    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor(
        [0, 0, 1.0], dtype=translation.dtype, device=translation.device
    )  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )

    # Apply quaternion for transformation from allocentric to egocentric.
    q_ego = quatmul_torch(q_allo_to_ego, q_allo)[
        :, 0
    ]  # Remove added Corner dimension here.
    return q_ego


def pose_from_pred(pred_rots, pred_transes, eps=1e-4, is_allo=True, is_train=True):
    if is_train:
        return pose_from_predictions_train(
            pred_rots, pred_transes, eps=eps, is_allo=is_allo
        )
    else:
        return pose_from_predictions_test(
            pred_rots, pred_transes, eps=eps, is_allo=is_allo
        )


def pose_from_predictions_test(pred_rots, pred_transes, eps=1e-4, is_allo=True):
    """NOTE: for test, non-differentiable"""
    translation = pred_transes

    # quat_allo = pred_quats / (torch.norm(pred_quats, dim=1, keepdim=True) + eps)
    # quat_ego = allocentric_to_egocentric_torch(translation, quat_allo, eps=eps)
    # use np since it is more accurate
    if pred_rots.shape[-1] == 4 and pred_rots.ndim == 2:
        pred_quats = pred_rots.detach().cpu().numpy()  # allo
        ego_rot_preds = np.zeros((pred_quats.shape[0], 3, 3), dtype=np.float32)
        for i in range(pred_quats.shape[0]):
            # try:
            if is_allo:
                # this allows unnormalized quat
                cur_ego_mat = allocentric_to_egocentric(
                    quat_trans_to_pose_m(
                        pred_quats[i], translation[i].detach().cpu().numpy()
                    ),
                    src_type="mat",
                    dst_type="mat",
                )[:3, :3]
            else:
                cur_ego_mat = quat_trans_to_pose_m(
                    pred_quats[i], translation[i].detach().cpu().numpy()
                )
            ego_rot_preds[i] = cur_ego_mat
            # except:

    # rot mat
    if pred_rots.shape[-1] == 3 and pred_rots.ndim == 3:
        pred_rots = pred_rots.detach().cpu().numpy()
        ego_rot_preds = np.zeros_like(pred_rots)
        for i in range(pred_rots.shape[0]):
            if is_allo:
                cur_ego_mat = allocentric_to_egocentric(
                    np.hstack(
                        [
                            pred_rots[i],
                            translation[i].detach().cpu().numpy().reshape(3, 1),
                        ]
                    ),
                    src_type="mat",
                    dst_type="mat",
                )[:3, :3]
            else:
                cur_ego_mat = pred_rots[i]
            ego_rot_preds[i] = cur_ego_mat
    return torch.from_np(ego_rot_preds), translation


def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    """
    Args:
        translation: Nx3
        rot_allo: Nx3x3
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor(
        [0, 0, 1.0], dtype=translation.dtype, device=translation.device
    )  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego


def pose_from_predictions_train(pred_rots, pred_transes, eps=1e-4, is_allo=True):
    """for train
    Args:
        pred_rots:
        pred_transes:
        eps:
        is_allo:
    Returns:
    """
    translation = pred_transes

    if pred_rots.ndim == 2 and pred_rots.shape[-1] == 4:
        pred_quats = pred_rots
        quat_allo = pred_quats / (torch.norm(pred_quats, dim=1, keepdim=True) + eps)
        if is_allo:
            quat_ego = allocentric_to_egocentric_torch(translation, quat_allo, eps=eps)
        else:
            quat_ego = quat_allo
        rot_ego = quat2mat_torch(quat_ego)
    if pred_rots.ndim == 3 and pred_rots.shape[-1] == 3:  # Nx3x3
        if is_allo:
            rot_ego = allo_to_ego_mat_torch(translation, pred_rots, eps=eps)
        else:
            rot_ego = pred_rots
    return rot_ego, translation


def pose_from_pred_centroid_z(
    pred_rots,
    pred_centroids,
    pred_z_vals,
    roi_cams,
    roi_centers,
    resize_ratios,
    roi_whs,
    eps=1e-4,
    is_allo=True,
    z_type="REL",
    is_train=True,
):
    if is_train:
        return pose_from_predictions_train_centroid(
            pred_rots,
            pred_centroids,
            pred_z_vals,
            roi_cams,
            roi_centers,
            resize_ratios,
            roi_whs,
            eps=eps,
            is_allo=is_allo,
            z_type=z_type,
        )
    else:
        return pose_from_predictions_test_centroid(
            pred_rots,
            pred_centroids,
            pred_z_vals,
            roi_cams,
            roi_centers,
            resize_ratios,
            roi_whs,
            eps=eps,
            is_allo=is_allo,
            z_type=z_type,
        )


def pose_from_predictions_test_centroid(
    pred_rots,
    pred_centroids,
    pred_z_vals,
    roi_cams,
    roi_centers,
    resize_ratios,
    roi_whs,
    eps=1e-4,
    is_allo=True,
    z_type="REL",
):
    """NOTE: for test, non-differentiable"""
    if roi_cams.dim() == 2:
        roi_cams.unsqueeze_(0)
    assert roi_cams.dim() == 3, roi_cams.dim()

    # absolute coords
    c = torch.stack(
        [
            (pred_centroids[:, 0] * roi_whs[:, 0]) + roi_centers[:, 0],
            (pred_centroids[:, 1] * roi_whs[:, 1]) + roi_centers[:, 1],
        ],
        dim=1,
    )

    cx = c[:, 0:1]  # [#roi, 1]
    cy = c[:, 1:2]  # [#roi, 1]

    # unnormalize regressed z
    if z_type == "abs":
        z = pred_z_vals
    elif z_type == "rel":
        # z_1 / z_2 = s_2 / s_1 ==> z_1 = s_2 / s_1 * z_2
        z = pred_z_vals * resize_ratios.view(-1, 1)
    else:
        raise ValueError(f"Unknown z_type: {z_type}")

    # backproject regressed centroid with regressed z
    """
    fx * tx + px * tz = z * cx
    fy * ty + py * tz = z * cy
    tz = z
    ==>
    fx * tx / tz = cx - px
    fy * ty / tz = cy - py
    ==>
    tx = (cx - px) * tz / fx
    ty = (cy - py) * tz / fy
    """
    translation = torch.cat(
        [
            z * (cx - roi_cams[:, 0:1, 2]) / roi_cams[:, 0:1, 0],
            z * (cy - roi_cams[:, 1:2, 2]) / roi_cams[:, 1:2, 1],
            z,
        ],
        dim=1,
    )

    # quat_allo = pred_quats / (torch.norm(pred_quats, dim=1, keepdim=True) + eps)
    # quat_ego = allocentric_to_egocentric_torch(translation, quat_allo, eps=eps)
    # use np since it is more accurate
    if pred_rots.shape[-1] == 4 and pred_rots.ndim == 2:
        pred_quats = pred_rots.detach().cpu().numpy()  # allo
        ego_rot_preds = np.zeros((pred_quats.shape[0], 3, 3), dtype=np.float32)
        for i in range(pred_quats.shape[0]):
            # try:
            if is_allo:
                # this allows unnormalized quat
                cur_ego_mat = allocentric_to_egocentric(
                    quat_trans_to_pose_m(
                        pred_quats[i], translation[i].detach().cpu().numpy()
                    ),
                    src_type="mat",
                    dst_type="mat",
                )[:3, :3]
            else:
                cur_ego_mat = quat_trans_to_pose_m(
                    pred_quats[i], translation[i].detach().cpu().numpy()
                )
            ego_rot_preds[i] = cur_ego_mat
            # except:

    # rot mat
    if pred_rots.shape[-1] == 3 and pred_rots.ndim == 3:
        pred_rots = pred_rots.detach().cpu().numpy()
        ego_rot_preds = np.zeros_like(pred_rots)
        for i in range(pred_rots.shape[0]):
            if is_allo:
                cur_ego_mat = allocentric_to_egocentric(
                    np.hstack(
                        [
                            pred_rots[i],
                            translation[i].detach().cpu().numpy().reshape(3, 1),
                        ]
                    ),
                    src_type="mat",
                    dst_type="mat",
                )[:3, :3]
            else:
                cur_ego_mat = pred_rots[i]
            ego_rot_preds[i] = cur_ego_mat
    return torch.from_np(ego_rot_preds), translation


def pose_from_predictions_train_centroid(
    pred_rots,
    pred_centroids,
    pred_z_vals,
    roi_cams,
    roi_centers,
    resize_ratios,
    roi_whs,
    eps=1e-4,
    is_allo=True,
    z_type="REL",
):
    """for train
    Args:
        pred_rots:
        pred_centroids:
        pred_z_vals: [B, 1]
        roi_cams: absolute cams
        roi_centers:
        roi_scales:
        roi_whs: (bw,bh) for bboxes
        eps:
        is_allo:
        z_type: REL | ABS | LOG | NEG_LOG
    Returns:
    """
    if roi_cams.dim() == 2:
        roi_cams.unsqueeze_(0)
    assert roi_cams.dim() == 3, roi_cams.dim()
    c = torch.stack(
        [
            (pred_centroids[:, 0] * roi_whs[:, 0]) + roi_centers[:, 0],
            (pred_centroids[:, 1] * roi_whs[:, 1]) + roi_centers[:, 1],
        ],
        dim=1,
    )

    cx = c[:, 0:1]  # [#roi, 1]
    cy = c[:, 1:2]  # [#roi, 1]

    # unnormalize regressed z
    if z_type == "abs":
        z = pred_z_vals
    elif z_type == "rel":
        # z_1 / z_2 = s_2 / s_1 ==> z_1 = s_2 / s_1 * z_2
        z = pred_z_vals * resize_ratios.view(-1, 1)
    else:
        raise ValueError(f"Unknown z_type: {z_type}")

    # backproject regressed centroid with regressed z
    """
    fx * tx + px * tz = z * cx
    fy * ty + py * tz = z * cy
    tz = z
    ==>
    fx * tx / tz = cx - px
    fy * ty / tz = cy - py
    ==>
    tx = (cx - px) * tz / fx
    ty = (cy - py) * tz / fy
    """

    translation = torch.cat(
        [
            z * (cx - roi_cams[:, 0:1, 2]) / roi_cams[:, 0:1, 0],
            z * (cy - roi_cams[:, 1:2, 2]) / roi_cams[:, 1:2, 1],
            z,
        ],
        dim=1,
    )

    if pred_rots.ndim == 2 and pred_rots.shape[-1] == 4:
        pred_quats = pred_rots
        quat_allo = pred_quats / (torch.norm(pred_quats, dim=1, keepdim=True) + eps)
        if is_allo:
            quat_ego = allocentric_to_egocentric_torch(translation, quat_allo, eps=eps)
        else:
            quat_ego = quat_allo
        rot_ego = quat2mat_torch(quat_ego)
    if pred_rots.ndim == 3 and pred_rots.shape[-1] == 3:  # Nx3x3
        if is_allo:
            rot_ego = allo_to_ego_mat_torch(translation, pred_rots, eps=eps)
        else:
            rot_ego = pred_rots
    return rot_ego, translation


def re(R_est, R_gt):
    """Rotational Error.
    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    rotation_diff = np.dot(R_est, R_gt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))

    return rd_deg


def get_closest_rot(rot_est, rot_gt, sym_info):
    """get the closest rot_gt given rot_est and sym_info.
    rot_est: ndarray
    rot_gt: ndarray
    sym_info: None or Kx3x3 ndarray, m2m
    """
    if sym_info is None:
        return rot_gt
    if isinstance(sym_info, torch.Tensor):
        sym_info = sym_info.cpu().numpy()
    if len(sym_info.shape) == 2:
        sym_info = sym_info.reshape((1, 3, 3))
    # find the closest rot_gt with smallest re
    r_err = re(rot_est, rot_gt)
    closest_rot_gt = rot_gt
    for i in range(sym_info.shape[0]):
        # R_gt_m2c x R_sym_m2m ==> R_gt_sym_m2c
        rot_gt_sym = rot_gt.dot(sym_info[i])
        cur_re = re(rot_est, rot_gt_sym)
        if cur_re < r_err:
            r_err = cur_re
            closest_rot_gt = rot_gt_sym

    return closest_rot_gt


def get_closest_rot_batch(pred_rots, gt_rots, sym_infos):
    """
    get closest gt_rots according to current predicted poses_est and sym_infos
    --------------------
    pred_rots: [B, 4] or [B, 3, 3]
    gt_rots: [B, 4] or [B, 3, 3]
    sym_infos: list [Kx3x3 or None],
        stores K rotations regarding symmetries, if not symmetric, None
    -----
    closest_gt_rots: [B, 3, 3]
    """
    batch_size = pred_rots.shape[0]
    device = pred_rots.device
    if pred_rots.shape[-1] == 4:
        pred_rots = quat2mat_torch(pred_rots[:, :4])
    if gt_rots.shape[-1] == 4:
        gt_rots = quat2mat_torch(gt_rots[:, :4])

    closest_gt_rots = gt_rots.clone().cpu().numpy()  # B,3,3

    for i in range(batch_size):
        closest_rot = get_closest_rot(
            pred_rots[i].detach().cpu().numpy(),
            gt_rots[i].cpu().numpy(),
            sym_infos[i],
        )
        # TODO: automatically detect rot_gt's format in PM_Loss to avoid converting multiple times
        closest_gt_rots[i] = closest_rot
    closest_gt_rots = torch.tensor(closest_gt_rots, device=device, dtype=gt_rots.dtype)
    return closest_gt_rots


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> np.allclose(np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2, np.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
