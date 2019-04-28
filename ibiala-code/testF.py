from submission import eightpoint
import numpy as np
import helper


project_data = np.load('../data/project_data2.npz')
matches = project_data['matches']
locs1 = project_data['locs1']
locs2 = project_data['locs2']
M1 = np.max(project_data['im1'].shape)
M2 = np.max(project_data['im2'].shape)
im1 = project_data['im1']
im2 = project_data['im2']
M = max(M1, M2)

match1 = locs1[matches[:, 0], 0:2]
match2 = locs2[matches[:, 1], 0:2]

F = eightpoint(match1, match2, M)
np.savez('../data/F2.npz', F=F)
helper.displayEpipolarF(im1, im2, F)

