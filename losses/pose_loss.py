


import torch.nn as nn

class PoseHead(nn.Module):
    def __init__(self, num_keypoints=33):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_keypoints * 2)
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(x.size(0), -1, 2)

JOINTS = [
    (11, 13, 15),  # left arm
    (12, 14, 16),  # right arm
    (23, 25, 27),  # left leg
    (24, 26, 28),  # right leg
    (11, 23, 25),  # left torso
    (12, 24, 26),  # right torso
]

import torch
import torch.nn.functional as F

def compute_joint_angles(kp, joints):
    """
    kp: [B, K, 2]
    return: [B, num_joints]
    """
    angles = []

    for (a, b, c) in joints:
        A = kp[:, a, :]
        B = kp[:, b, :]
        C = kp[:, c, :]

        BA = A - B
        BC = C - B

        BA = F.normalize(BA, dim=1)
        BC = F.normalize(BC, dim=1)

        cos_angle = torch.sum(BA * BC, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

        angle = torch.acos(cos_angle)
        angles.append(angle)

    return torch.stack(angles, dim=1)

def pose_angle_loss(real_kp, pred_kp, joints, alpha=1.0):
    """
    Angle-based geometric supervision
    """
    # Normalize (translation invariant)
    real_kp = real_kp - real_kp.mean(dim=1, keepdim=True)
    pred_kp = pred_kp - pred_kp.mean(dim=1, keepdim=True)

    real_angles = compute_joint_angles(real_kp, joints)
    pred_angles = compute_joint_angles(pred_kp, joints)
     # convert to cosine space (stable)
    real_cos = torch.cos(real_angles)
    pred_cos = torch.cos(pred_angles)

    return alpha * F.mse_loss(pred_cos, real_cos)

  
