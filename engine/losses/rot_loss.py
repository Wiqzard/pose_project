import torch
import torch.nn as nn


def angular_distance(r1, r2, reduction="mean"):
    """https://math.stackexchange.com/questions/90081/quaternion-distance
    https.
    ://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tool
    s.py.
    1 - <q1, q2>^2  <==> (1-cos(theta)) / 2
    """
    assert r1.shape == r2.shape
    if r1.shape[-1] == 4:
        return angular_distance_quat(r1, r2, reduction=reduction)
    else:
        return angular_distance_rot(r1, r2)  # , reduction=reduction)


def angular_distance_quat(pred_q, gt_q, reduction="mean"):
    dist = 1 - torch.pow(torch.bmm(pred_q.view(-1, 1, 4), gt_q.view(-1, 4, 1)), 2)
    if reduction == "mean":
        return dist.mean()
    elif reduction == "sum":
        return dist.sum()
    else:
        return dist


# def angular_distance_rot(m1, m2, reduction="mean"):
#    m = torch.bmm(m1, m2.transpose(1, 2))  # b*3*3
#    m_trace = torch.einsum("bii->b", m)  # batch trace
#    cos = (m_trace - 1) / 2  # [-1, 1]
#    # eps = 1e-6
#    # cos = torch.clamp(cos, -1+eps, 1-eps)  # avoid nan
#    # theta = torch.acos(cos)
#    dist = (1 - cos) / 2  # [0, 1]
#    if reduction == "mean":
#        return dist.mean()
#    elif reduction == "sum":
#        return dist.sum()
#    else:
#        return dist


def angular_distance_rot(pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular loss between predicted and ground truth batched rotation matrices.
    Args:
        pred_rot (torch.Tensor): The predicted batch of 3x3 rotation matrices with shape (batch_size, 3, 3).
        gt_rot (torch.Tensor): The ground truth batch of 3x3 rotation matrices with shape (batch_size, 3, 3).
    Returns:
        torch.Tensor: The mean angular loss.
    """
    rot_mul = torch.matmul(gt_rot, torch.transpose(pred_rot, 1, 2))
    cos_angle = (torch.einsum("bii->b", rot_mul) - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle_rad = torch.acos(cos_angle)
    angle_deg = torch.rad2deg(angle_rad)

    return angle_deg #.mean()


def rotation_matrix_to_angle_axis(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix to an angle-axis representation.
    Args:
        matrix (torch.Tensor): A 3x3 rotation matrix.
    Returns:
        torch.Tensor: A 3D vector representing the rotation axis scaled by the rotation angle.
    """
    cos_angle = (torch.trace(matrix) - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle = torch.acos(cos_angle)
    axis = torch.zeros(3)

    if angle < 1e-3:
        return axis

    sin_angle = torch.sin(angle)
    axis[0] = (matrix[2, 1] - matrix[1, 2]) / (2 * sin_angle)
    axis[1] = (matrix[0, 2] - matrix[2, 0]) / (2 * sin_angle)
    axis[2] = (matrix[1, 0] - matrix[0, 1]) / (2 * sin_angle)

    return axis * angle


def angular_distance_loss(matrix1: torch.Tensor, matrix2: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular distance loss between two 3x3 rotation matrices.
    Args:
        matrix1 (torch.Tensor): The first 3x3 rotation matrix.
        matrix2 (torch.Tensor): The second 3x3 rotation matrix.
    Returns:
        torch.Tensor: The scalar angular distance loss.
    """
    angle_axis1 = rotation_matrix_to_angle_axis(matrix1)
    angle_axis2 = rotation_matrix_to_angle_axis(matrix2)

    loss = nn.MSELoss()(angle_axis1, angle_axis2)
    return loss


def rot_l2_loss(m1, m2):
    error = torch.pow(m1 - m2, 2).mean()  # batch
    return error
