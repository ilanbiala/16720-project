from submission import eightpoint
import numpy as np
from helper import _objective_F
import helper
from BRIEF import plotMatches


project_data = np.load('../data/project_data.npz')
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
n = match1.shape[0]
s = 16
count = 0
num_iter = 5000
min_r = 10000000
F = None
best_index = None

for i in range(num_iter):
    index = np.random.randint(0, n, size=s)
    p1 = match1[index]
    p2 = match2[index]
    unscaledF = eightpoint(p1, p2, M)
    r = _objective_F(unscaledF, p1, p2)
    if r < min_r:
        min_r = r
        F = unscaledF
        best_index = index

plotMatches(im1, im2, matches[best_index], locs1, locs2)
helper.displayEpipolarF(im1, im2, F)
