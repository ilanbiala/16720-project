import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    pano_im = cv2.warpPerspective(im2, H2to1, (1500, 900))

    # Add in original image
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            for k in range(im1.shape[2]):
                pano_im[i, j, k] = max(im1[i, j, k], pano_im[i, j, k])

    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given
    homography matrix without clipping.
    '''
    x_corners = np.array([0, 0, im2.shape[1] - 1, im2.shape[1] - 1]).reshape(1,4)
    y_corners = np.array([0, im2.shape[0] - 1, im2.shape[0] - 1, 0]).reshape(1,4)
    corners = np.concatenate((x_corners, y_corners, np.ones((1, 4))), axis=0)
    # Transform coordinates into 3rd reference frame
    transformed_coordinates = np.matmul(H2to1, corners)
    transformed_norm = np.divide(transformed_coordinates, transformed_coordinates[2,:])

    x1_min = 0
    y1_min = 0
    x1_max = im1.shape[1] - 1
    y1_max = im1.shape[0] - 1
    x2_min = np.min(transformed_norm[0])
    y2_min = np.min(transformed_norm[1])
    x2_max = np.max(transformed_norm[0])
    y2_max = np.max(transformed_norm[1])

    x_min = min(x1_min, x2_min)
    y_min = min(y1_min, y2_min)
    x_max = max(x1_max, x2_max)
    y_max = max(y1_max, y2_max)

    final_image_size = (int(x_max - x_min), int(y_max - y_min))
    x_offset = int(abs(x_min))
    y_offset = int(abs(y_min))
    M = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]], dtype='f')

    # Warp both into 3rd reference frame
    warped_im1 = cv2.warpPerspective(im1, M, final_image_size)
    warped_im2 = cv2.warpPerspective(im2, M.dot(H2to1), final_image_size)
    pano_im = np.maximum(warped_im1, warped_im2)
    return pano_im


def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    print(im1.shape)
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

    pano_im = imageStitching(im1, im2, H2to1)
    np.save('../results/q6_1.npy', H2to1)
    cv2.imwrite('../results/q6_1.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)

    pano_im = imageStitching_noClip(im1, im2, H2to1)
    print(H2to1)
    cv2.imwrite('../results/q6_2.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pano_im = generatePanorama(im1, im2)
    np.save('../results/q6_3.jpg', pano_im)
