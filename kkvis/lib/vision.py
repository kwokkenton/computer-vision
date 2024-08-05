import numpy as np


def make_intrinsics(fx, fy, cx, cy, s=0):
    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 1] = s
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def form_transf(R, t):
    # Equivalent to cv2.hconcat([R, t])
    # Which is slower
    pose = np.eye(4)
    pose[0:3, 0:3] = R
    pose[:3, 3] = t

    return pose
