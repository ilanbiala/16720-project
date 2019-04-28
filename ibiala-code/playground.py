import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

import helper
import submission as sub

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

correspondences = np.load('../data/some_corresp.npz')
intrinsics = np.load('../data/intrinsics.npz')

pts1 = correspondences['pts1']
pts2 = correspondences['pts2']
M = max(im1.shape)

K1 = intrinsics['K1']
K2 = intrinsics['K2']

#
project_data = np.load('../data/project_data.npz')
im1 = project_data['im1']
im2 = project_data['im2']
locs1 = project_data['locs1']
locs2 = project_data['locs2']
matches = project_data['matches']
pts1 = locs1[matches[:, 0]][:, 0:2]
pts2 = locs2[matches[:, 1]][:, 0:2]
M = max(im1.shape)
#

F = sub.eightpoint(pts1, pts2, M)

# Q2.1
print(f'F:\n{F}')
np.savez('q2_1.npz', M=M, F=F)
helper.displayEpipolarF(im1, im2, F)

import sys; sys.exit()

# rand_indices = np.random.choice(np.arange(pts1.shape[0]), 7)
rand_indices = np.array([108, 45, 3, 102, 84, 22, 71]) # Chosen empirically
seven_pts1 = pts1[rand_indices]
seven_pts2 = pts2[rand_indices]

M = max(im1.shape)
Farray = sub.sevenpoint(seven_pts1, seven_pts2, M)
print(rand_indices)

# Q2.2
# for i in range(len(Farray)): 3rd F is best
print(f'Farray[2]:\n{Farray[2]}')
helper.displayEpipolarF(im1, im2, Farray[2])
np.savez('q2_2.npz', F=Farray[2], M=M, pts1=seven_pts1, pts2=seven_pts2)

# Q3.1
E = sub.essentialMatrix(F, K1, K2)
print(f'E:\n{E}')

# Q4.1
helper.epipolarMatchGUI(im1, im2, F)
np.savez('q4_1.npz', F=F, pts1=pts1, pts2=pts2)
