'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
from submission import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def main():
    pts = np.load('../data/templeCoords.npz')
    x1 = pts['x1']
    y1 = pts['y1']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    F = np.load('../data/F_temple.npz')['F']

    pts2 = []
    for i in range(x1.shape[0]):
        pt2 = epipolarCorrespondence(im1, im2, F, x1[i][0], y1[i][0])
        pts2.append(pt2)

    pts2 = np.array(pts2)
    pts1 = np.concatenate((x1, y1), axis=1)

    K = np.load('../data/intrinsics.npz')
    K1 = K['K1']
    K2 = K['K2']
    E = essentialMatrix(F, K1, K2)
    M2s = helper.camera2(E)
    M1 = np.concatenate((np.identity(3), np.zeros((3, 1))), axis=1)
    C1 = np.matmul(K1, M1)

    correct_M2 = None
    correct_w = None
    for i in range(M2s.shape[2]):
        M2 = M2s[:, :, i]
        C2 = np.matmul(K2, M2)
        w, err = triangulate(C1, pts1, C2, pts2)
        if np.all(w[:, 2] > 0):
            correct_M2 = M2
            correct_w = w
            print(err)

    C2 = np.matmul(K2, correct_M2)

    np.savez('q4_2.npz', M1=M1, M2=correct_M2, C1=C1, C2=C2, F=F)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xmin, xmax = np.min(correct_w[:, 0]), np.max(correct_w[:, 0])
    ymin, ymax = np.min(correct_w[:, 1]), np.max(correct_w[:, 1])
    zmin, zmax = np.min(correct_w[:, 2]), np.max(correct_w[:, 2])

    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)

    ax.scatter(correct_w[:, 0], correct_w[:, 1], correct_w[:, 2], c='b', marker='o')
    plt.show()


if __name__ == '__main__':
    main()
