from submission import *
from helper import *

import cv2
import numpy as np
import matplotlib.pyplot as plt
from BRIEF import plotMatches
from mpl_toolkits.mplot3d import Axes3D


def plotCorresponds(im1, im2, pts1, pts2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(pts1.shape[0]):
        pt1 = pts1[i]
        pt2 = pts2[i]
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r')
        plt.plot(x,y,'g.')
    plt.show()
'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
coords = np.load('../data/coords_palace.npz')
# intrinsics = np.load('../data/intrinsics.npz')
intrinsics = np.eye(3)
# intrinsics = 1.0e3*np.array([
#     [3.8324, 0, 0], [0, 3.8178, 0],
#     [1.6911, 2.2214, 0.0010]])
W = int(1920/4)
H = int(1080/4)
intrinsics = 1.0e3*np.array([
    [0.7*W , 0,  W/2],
    [0, 0.7*W ,  H/2],
    [0, 0, 1]])
data = np.load('../data/project_palace.npz')

im1 = data['im1']
im2 = data['im2']
matches = data['matches']
locs1 = data['locs1']
locs2 = data['locs2']


x1 = coords['x1']
y1 = coords['y1']
x1 = np.expand_dims(x1, 1)
y1 = np.expand_dims(y1, 1)
num_points = x1.shape[0]

x2 = np.zeros((num_points, 1))
y2 = np.zeros((num_points, 1))

K1 = intrinsics
K2 = intrinsics

M = max(im1.shape)
num_points = x1.shape[0]

F = np.load('../data/F_palace.npz')['F']
# displayEpipolarF(im1, im2, F)
E = essentialMatrix(F, K1, K2)

for i in range(num_points):
    x2[i], y2[i] = epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
points1 = np.append(x1, y1, axis=1)
points2 = np.append(x2, y2, axis=1)

plotCorresponds(im1, im2, points1, points2)

M1 = np.array([[1.0, 0, 0, 0],
               [0, 1.0, 0, 0],
               [0, 0, 1.0, 0]])
M2s = camera2(E)

C1 = np.dot(K1, M1)

min_error = 1e12
min_C2 = 0
min_P = 0
min_index = 0

for i in range(4):
    C2 = np.dot(K2, M2s[:, :, i])
    P, error = triangulate(C1, points1, C2, points2)
    if error < min_error:
        if np.min(P[:, 2] >= 0):
            print("Found new minimum error...")
            min_error = np.copy(error)
            min_index = np.copy(i)
            min_C2 = np.copy(C2)
            min_P = np.copy(P)
np.savez("q4_2.npz", F=F, M1=M1, M2=M2s[:, :, min_index], C1=C1, C2=min_C2)
print(f"Final min_error={min_error}")
P = np.copy(min_P)
# print(f'P: {P}')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xmin, xmax = np.min(P[:, 0]), np.max(P[:, 0])
ymin, ymax = np.min(P[:, 1]), np.max(P[:, 1])
zmin, zmax = np.min(P[:, 2]), np.max(P[:, 2])

ax.set_xlim3d(xmin, xmax)
ax.set_ylim3d(ymin, ymax)
ax.set_zlim3d(zmin, zmax)

ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
plt.show()
