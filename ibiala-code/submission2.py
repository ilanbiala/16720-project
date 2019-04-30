"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
from scipy.ndimage.filters import gaussian_filter
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''


def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1/M
    pts2 = pts2/M
    N = pts1.shape[0]

    U = np.zeros((N, 9))
    U[:, 0] = pts2[:, 0]*pts1[:, 0]
    U[:, 1] = pts2[:, 0]*pts1[:, 1]
    U[:, 2] = pts2[:, 0]
    U[:, 3] = pts2[:, 1]*pts1[:, 0]
    U[:, 4] = pts2[:, 1]*pts1[:, 1]
    U[:, 5] = pts2[:, 1]
    U[:, 6] = pts1[:, 0]
    U[:, 7] = pts1[:, 1]
    U[:, 8] = np.ones(N)

    m = np.matmul(U.T, U)
    w, v = np.linalg.eigh(m)
    index = np.argmin(w)
    F = v[:, index].reshape(3, 3)
    u, s, v = np.linalg.svd(F)
    # force F to be singular
    F = np.matmul(np.matmul(u, np.diag([s[0], s[1], 0])), v)
    F = helper.refineF(F, pts1, pts2)
    # unscale the fundamental matrix
    T = np.diag([1 / M, 1 / M, 1])
    F = np.matmul(np.matmul(T.T, F), T)
    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''


def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1 = pts1 / M
    pts2 = pts2 / M
    N = pts1.shape[0]

    U = np.zeros((N, 9))
    U[:, 0] = pts2[:, 0] * pts1[:, 0]
    U[:, 1] = pts2[:, 0] * pts1[:, 1]
    U[:, 2] = pts2[:, 0]
    U[:, 3] = pts2[:, 1] * pts1[:, 0]
    U[:, 4] = pts2[:, 1] * pts1[:, 1]
    U[:, 5] = pts2[:, 1]
    U[:, 6] = pts1[:, 0]
    U[:, 7] = pts1[:, 1]
    U[:, 8] = np.ones(N)

    _, _, v = np.linalg.svd(U)
    F1 = v.T[:, -1].reshape(3, 3)
    F2 = v.T[:, -2].reshape(3, 3)

    x = lambda a: np.linalg.det(a*F1 + (1-a)*F2)
    # a3x^3 + a2x^2 + a1x + a0
    # x(1) + x(-1) = 2a2 + 2a0
    # 2x(1) - 2x(-1) = 4a3 + 4a1
    # x(2) - x(-2) = 16a3 + 4a1
    a0 = x(0)
    a2 = (x(1) + x(-1))/2 - a0
    a3 = (x(2) - x(-2) - 2*x(1) + 2*x(-1))/12
    a1 = (x(1) - x(-1))/2 - a3

    roots = np.roots([a3, a2, a1, a0])
    Farray = []
    for root in roots:
        F = np.real(root)*F1 + (1-np.real(root))*F2
        u, s, v = np.linalg.svd(F)
        # force F to be singular
        F = np.matmul(np.matmul(u, np.diag([s[0], s[1], 0])), v)
        # unscale the fundamental matrix
        T = np.diag([1 / M, 1 / M, 1])
        F = np.matmul(np.matmul(T.T, F), T)
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
    # Replace pass by your implementation
    return np.matmul(np.matmul(K1.T, F), K2)


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
    # Replace pass by your implementation
    N = pts1.shape[0]
    coor = []
    err = 0
    for i in range(N):
        tmp1 = np.matmul(np.expand_dims(pts1[i], 1), np.expand_dims(C1[-1], 0)) - C1[0: 2]
        tmp2 = np.matmul(np.expand_dims(pts2[i], 1), np.expand_dims(C2[-1], 0)) - C2[0: 2]
        Ai = np.concatenate((tmp1, tmp2))
        m = np.matmul(Ai.T, Ai)
        w, v = np.linalg.eigh(m)
        index = np.argmin(w)
        wi = v[:, index]
        wi /= wi[3]
        coor.append(wi)
        pt1_reprojected = np.matmul(C1, wi)
        pt1_reprojected /= pt1_reprojected[2]
        pt2_reprojected = np.matmul(C2, wi)
        pt2_reprojected /= pt2_reprojected[2]

        err += np.linalg.norm(pt1_reprojected[0: 2] - pts1[i]) + np.linalg.norm(pt2_reprojected[0: 2] - pts2[i])

    return np.array(coor), err


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


def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    window_size = 7
    pt1 = np.expand_dims(np.array([x1, y1, 1]), 1)
    line = np.squeeze(np.matmul(pt1.T, F))

    patch1 = np.copy(im1[y1-window_size: y1+window_size, x1-window_size: x1+window_size])
    y = np.arange(window_size, im1.shape[0]-window_size)
    x = (-line[2]-line[1]*y)/line[0]
    mask = np.logical_and(x < im1.shape[1] - window_size, x >= window_size)
    pts = np.array([x[mask], y[mask]]).T

    error_min = float("inf")
    pt2 = pts[0]

    for pt in pts.astype(np.int):
        if np.linalg.norm(np.squeeze(pt1[0: 2]) - pt) < 50:
            patch2 = np.copy(im2[pt[1]-window_size: pt[1]+window_size, pt[0]-window_size: pt[0]+window_size])

            error = np.sqrt((patch1 - patch2)**2)

            error_filtered = np.sum(gaussian_filter(error, sigma=1.0))
            if error_filtered < error_min:
                error_min = error_filtered
                pt2 = pt

    return pt2[0], pt2[1]


