import cv2
import matplotlib.pyplot as plt
import numpy as np

from kkvis.lib.vision import form_transf


def drawlines(img1, img2, lines, pts1, pts2):
    # Draws lines on image
    # Modified from https://docs.opencv.org/4.x/da/de9/tutorial_py_
    # epipolar_geometry.html
    if len(img1.shape) == 3 or len(img2.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Assume sizes are the same
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Generate list of colours

    for r, pt1, pt2 in zip(lines, pts1, pts2):

        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)
    return img1, img2


def plot_epilines(im1, im2, pts1, pts2, F):
    # Find epilines corresponding to points
    # in right image (second image) and
    # drawing its lines on left image
    linesLeft = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    linesLeft = linesLeft.reshape(-1, 3)
    img5, img6 = drawlines(im1, im2, linesLeft, pts1, pts2)

    # Find epilines corresponding to
    # points in left image (first image) and
    # drawing its lines on right image
    linesRight = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)

    linesRight = linesRight.reshape(-1, 3)

    img3, img4 = drawlines(im2, im1, linesRight, pts2, pts1)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()
    return


def to_homogeneous(x):
    b = np.ones(len(x))
    return np.concatenate((x, b[:, None]), axis=1)


def longuet_higgins(q0, F, q1):
    return np.einsum("ij, jk, ik -> i", to_homogeneous(q1), F, to_homogeneous(q0))


def sum_z_cal_relative_scale(R, t, q1, q2, K1, K2):
    """_summary_

    Args:
        R (_type_): _description_
        t (_type_): _description_
        q1 (_type_): _description_
        q2 (_type_): _description_
        K1 (_type_): _description_
        K2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Get the transformation matrix
    T = form_transf(R, t)
    # Make the projection matrices, taking the pose of camera 1 to be at the origin
    P_ref = np.concatenate((K1, np.zeros((3, 1))), axis=1)
    P_test = np.matmul(np.concatenate((K2, np.zeros((3, 1))), axis=1), T)

    # Triangulate the 3D points given the camera matrices and the image coordinates
    # to give homogeneous coordinates
    hom_Q1 = cv2.triangulatePoints(P_ref, P_test, q1.T, q2.T)
    # Determine the
    hom_Q2 = np.matmul(T, hom_Q1)

    # Un-homogenize
    uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
    uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

    # Find the number of points that has positive z coordinate in both cameras
    sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
    sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

    # Form point pairs and calculate the relative scale
    relative_scale = np.mean(
        np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)
        / np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)
    )
    return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale


def decomp_essential_mat(E, q1, q2, K1, K2):
    """Decompose the Essential matrix into rotation and translation.
    Verify that the solution is correct by projecting using K1 and K2

    TODO: change this to Hartley and Zisserman's solution

    From https://www.youtube.com/watch?v=N451VeA8XRA

    Args:
        E (_type_): Essential matrix
        q1 (_type_): The image coordinates of keypoint matches position in image 1
        q2 (_type_): The image coordinates of keypoint matches position in image 2

    Returns:
        right_pair (list): Contains the rotation matrix and translation vector
    """

    # Decompose the essential matrix
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = np.squeeze(t)

    # Make a list of the different possible pairs
    pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

    # Check which solution is the physical one (in front of the two cameras)
    z_sums = []
    relative_scales = []
    for R, t in pairs:
        z_sum, scale = sum_z_cal_relative_scale(R, t, q1, q2, K1, K2)
        z_sums.append(z_sum)
        relative_scales.append(scale)

    # Select the pair there has the most points with positive z coordinate
    right_pair_idx = np.argmax(z_sums)
    right_pair = pairs[right_pair_idx]
    relative_scale = relative_scales[right_pair_idx]
    R1, t = right_pair
    t = t * relative_scale

    return [R1, t]

# TODO make
