import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)

    A = np.zeros((2 * p1.shape[1], 9))
    for i in range(p1.shape[1]):
        x = p1[0, i]
        y = p1[1, i]
        x_prime = p2[0, i]
        v_prime = p2[1, i]

        A[2 * i] = [0, 0, 0, -x_prime, -v_prime, -1, y * x_prime, y * v_prime, y]
        A[2 * i + 1] = [x_prime, v_prime, 1, 0, 0, 0, -x * x_prime, -x * v_prime, -x]

    vals, vectors = np.linalg.eigh(A.T.dot(A))
    H2to1 = vectors[:, 0].reshape((3, 3))
    return H2to1


# def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
#     '''
#     Returns the best homography by computing the best set of matches using
#     RANSAC
#     INPUTS
#         locs1 and locs2 - matrices specifying point locations in each of the images
#         matches - matrix specifying matches between these two sets of point locations
#         nIter - number of iterations to run RANSAC
#         tol - tolerance value for considering a point to be an inlier

#     OUTPUTS
#         bestH - homography matrix with the most inliers found during RANSAC
#     '''

#     '''
#     Draw set with minimum number of correspondences
#     Fit Homography to the set
#     Count the number d of correspondences that are closer than t to the fitted homography
#     If d > dmin, recompute fit error using all the correspondences
#     Return best fit found
#     '''
#     print('matches size', matches.shape)

#     locs1_2d = locs1[:, :2].T
#     locs2_2d = locs2[:, :2].T
#     locs1_homogeneous_start = np.append(locs1_2d, np.ones((1, locs1_2d.shape[1])), axis=0)
#     locs2_homogeneous_start = np.append(locs2_2d, np.ones((1, locs2_2d.shape[1])), axis=0)

#     most_inliers = np.empty((0, matches.shape[1]))
#     for iter_i in range(num_iter):
#         rand_correspondence_choices = np.random.choice(matches.shape[0], size=4, replace=False)

#         p1 = locs1_2d[:, matches[rand_correspondence_choices][:, 0]]
#         p2 = locs2_2d[:, matches[rand_correspondence_choices][:, 1]]

#         bestH = computeH(p1, p2)
#         # Compute homogeneous coordinates
#         locs2_homogeneous = bestH.dot(locs2_homogeneous_start)
#         # Normalize properly into other coordinate system
#         locs2_homogeneous = locs2_homogeneous / locs2_homogeneous[2]

#         distances = np.linalg.norm(locs1_homogeneous_start[:, matches[:, 0]] - locs2_homogeneous[:, matches[:, 1]], axis=0)
#         inliers = distances <= tol
#         num_inliers = inliers.sum()
#         if num_inliers > np.sum(most_inliers):
#             most_inliers = inliers

#         # Count correspondences, normalizing the new coordinates since they've been multiplied by the homography
#         # Check against all locs1, locs2 to see how many points are within tol distance
#         # keep repeating and keeping track of most number of inliers

#     print('inliers', np.sum(most_inliers))
#     return bestH

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''

    '''
    Draw set with minimum number of correspondences
    Fit Homography to the set
    Count the number d of correspondences that are closer than t to the fitted homography
    If d > dmin, recompute fit error using all the correspondences
    Return best fit found
    '''
    print('Matches:', matches.shape)
    most_num_inliers = -1
    bestH = np.zeros((3,3))

    locs1_matches = locs1[matches[:, 0], 0:2]
    locs2_matches = locs2[matches[:, 1], 0:2]
    locs1_matches_homogeneous = np.append(locs1_matches.T, np.ones((1, matches.shape[0])), axis=0)
    locs2_matches_homogeneous = np.append(locs2_matches.T, np.ones((1, matches.shape[0])), axis=0)

    locs1_2d = locs1[:, :2].T
    locs2_2d = locs2[:, :2].T

    for i in range(num_iter):
        rand_correspondence_choices = np.random.choice(matches.shape[0], size=4, replace=False)
        p1 = locs1_2d[:, matches[rand_correspondence_choices][:, 0]]
        p2 = locs2_2d[:, matches[rand_correspondence_choices][:, 1]]

        H = computeH(p1, p2)
        locs2_homogeneous = np.matmul(H, locs2_matches_homogeneous)
        locs2_homogeneous_norm = np.divide(locs2_homogeneous, locs2_homogeneous[2, :])

        # Compute number of inliers
        distances = np.linalg.norm(locs1_matches_homogeneous - locs2_homogeneous_norm, axis=0)
        inliers = distances <= tol
        num_inliers = inliers.sum()
        if num_inliers > most_num_inliers:
            bestH = H
            most_num_inliers = num_inliers

    print("Most inliers: ", most_num_inliers)
    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
