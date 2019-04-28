"""
Homework4.
Replace 'pass' by your implementation.
"""
import math

import numpy as np
import numpy.matlib
import scipy
import scipy.ndimage
# from scipy.ndimage.filters import gaussian_filter

import helper

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    num_points = pts1.shape[0]
    scaled_pts1 = pts1 / M
    scaled_pts2 = pts2 / M

    correspondence_matrix = np.zeros((num_points, 9))
    correspondence_matrix[:, 0] = scaled_pts2[:, 0] * pts1[:, 0]
    correspondence_matrix[:, 1] = scaled_pts2[:, 0] * pts1[:, 1]
    correspondence_matrix[:, 2] = scaled_pts2[:, 0]
    correspondence_matrix[:, 3] = scaled_pts2[:, 1] * pts1[:, 0]
    correspondence_matrix[:, 4] = scaled_pts2[:, 1] * pts1[:, 1]
    correspondence_matrix[:, 5] = scaled_pts2[:, 1]
    correspondence_matrix[:, 6] = scaled_pts1[:, 0]
    correspondence_matrix[:, 7] = scaled_pts1[:, 1]
    correspondence_matrix[:, 8] = np.ones(num_points)

    U, S, V = np.linalg.svd(correspondence_matrix)
    F = V.T[:, -1].reshape((3, 3))
    UF, SF, VF = np.linalg.svd(F)

    d1, d2, d3 = SF[0], SF[1], 0
    S = np.diag([d1, d2, d3])
    estimatedF = UF @ S @ VF

    refinedF = helper.refineF(estimatedF, scaled_pts1, scaled_pts2)
    T = np.diag([1.0/M, 1.0/M, 1.0])
    unscaledF = T.T @ refinedF @ T

    return unscaledF


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    assert(pts1.shape == (7, 2))
    assert(pts2.shape == (7, 2))

    num_points = pts1.shape[0]
    scaled_pts1 = pts1 / M
    scaled_pts2 = pts2 / M

    correspondence_matrix = np.zeros((num_points, 9))
    correspondence_matrix[:, 0] = scaled_pts2[:, 0] * pts1[:, 0]
    correspondence_matrix[:, 1] = scaled_pts2[:, 0] * pts1[:, 1]
    correspondence_matrix[:, 2] = scaled_pts2[:, 0]
    correspondence_matrix[:, 3] = scaled_pts2[:, 1] * pts1[:, 0]
    correspondence_matrix[:, 4] = scaled_pts2[:, 1] * pts1[:, 1]
    correspondence_matrix[:, 5] = scaled_pts2[:, 1]
    correspondence_matrix[:, 6] = scaled_pts1[:, 0]
    correspondence_matrix[:, 7] = scaled_pts1[:, 1]
    correspondence_matrix[:, 8] = np.ones(num_points)

    U, S, V = np.linalg.svd(correspondence_matrix)
    F1 = V.T[:, -1].reshape((3, 3))
    F2 = V.T[:, -2].reshape((3, 3))

    fun = lambda a: np.linalg.det(a * F1 + (1 - a) * F2)

    a0 = fun(0)
    a1 = 2.0 * (fun(1) - fun(-1)) / 3 - (fun(2) - fun(-2)) / 12
    a2 = 0.5 * fun(1) + 0.5 * fun(-1) - fun(0)
    a3 = fun(1) - a0 - a1 - a2
    roots = np.roots(np.array([a3, a2, a1, a0]))

    T = np.diag([1.0/M, 1.0/M, 1.0])
    Farray = []

    for alpha in roots:
        F = F1 * float(np.real(alpha)) + F2 * (1 - float(np.real(alpha)))
        U, S, V = np.linalg.svd(F)

        ss = np.diag([S[0], S[1], S[2]])
        F = U @ ss @ V
        F = T.T @ F @ T
        Farray.append(F)

    return Farray


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return K2.T @ F @ K1


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    num_points = pts1.shape[0]
    P = np.zeros((num_points, 3))
    A = np.zeros((4, 4))

    for i in range(num_points):
        A[0] = pts1[i, 0] * C1[2] - C1[0]
        A[1] = pts1[i, 1] * C1[2] - C1[1]
        A[2] = pts2[i, 0] * C2[2] - C2[0]
        A[3] = pts2[i, 1] * C2[2] - C2[1]
        U, S, V = np.linalg.svd(A)

        p = V.T[:, -1]
        P[i] = np.divide(p[0:3], p[3])

    P_homo = np.append(P, np.ones((P.shape[0], 1)), axis=1).T

    p1_reprojected = np.matmul(C1, P_homo)
    p2_reprojected = np.matmul(C2, P_homo)
    p1_normalized = np.zeros((2, num_points))
    p2_normalized = np.zeros((2, num_points))

    p1_normalized[0] = p1_reprojected[0] / p1_reprojected[2]
    p1_normalized[1] = p1_reprojected[1] / p1_reprojected[2]
    p2_normalized[0] = p2_reprojected[0] / p2_reprojected[2]
    p2_normalized[1] = p2_reprojected[1] / p2_reprojected[2]

    p1_normalized = p1_normalized.T
    p2_normalized = p2_normalized.T

    p1_error = (p1_normalized - pts1)[:, 0] * (p1_normalized - pts1)[:, 0] + (p1_normalized - pts1)[:, 1] * (p1_normalized - pts1)[:, 1]
    p2_error = (p2_normalized - pts2)[:, 0] * (p2_normalized - pts2)[:, 0] + (p2_normalized - pts2)[:, 1] * (p2_normalized - pts2)[:, 1]

    error = np.sum(p1_error) + np.sum(p2_error)
    return P, error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1, window=10):
    im1 = np.copy(im1.astype(float))
    im2 = np.copy(im2.astype(float))

    point = np.array([[x1], [y1], [1]])
    l = np.dot(F, point)
    l = l / np.linalg.norm(l)

    pts_xy = np.empty((0, 2))
    pts_yx = np.empty((0, 2))

    if l[0] != 0:
        for y in range(window, im2.shape[0] - window):
            x = math.floor(-1.0 * (l[1] * y + l[2]) / l[0])
            if x >= window and x <= im2.shape[1] - window:
                pts_yx = np.append(pts_yx, np.array([x, y]).reshape(1, 2), axis=0)
    else:
        for x in range(window, im2.shape[1] - window):
            y = math.floor(-1.0 * (l[0] * x + l[2]) / l[1])
            if y >= window and y <= im2.shape[0] - window:
                pts_xy = np.append(pts_xy, np.array([x, y]).reshape(1, 2), axis=0)

    pts = pts_yx

    patch1 = im1[int(y1 - window + 1):int(y1 + window), int(x1 - window + 1):int(x1 + window), :]

    min_error = 1e12
    min_index = 0

    for i in range(pts.shape[0]):
        x2 = pts[i, 0]
        y2 = pts[i, 1]
        if math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) < 50:
            patch2 = im2[int(y2 - window + 1):int(y2 + window), int(x2 - window + 1):int(x2 + window), :]
            error = patch1 - patch2
            error_filtered = np.sum(scipy.ndimage.gaussian_filter(error, sigma=1.0))
            if error_filtered < min_error:
                min_error = error_filtered
                min_index = i

    x2 = pts[min_index, 0]
    y2 = pts[min_index, 1]
    return x2, y2
