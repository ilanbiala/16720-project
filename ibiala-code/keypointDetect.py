import numpy as np
import cv2
import scipy
import scipy.signal

def createGaussianPyramid(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = np.ndarray((gaussian_pyramid.shape[0], gaussian_pyramid.shape[1], gaussian_pyramid.shape[2] - 1))
    for i in range(1, len(levels)):
        DoG_pyramid[:, :, i - 1] = (gaussian_pyramid[:, :, i] - gaussian_pyramid[:, :, i - 1])
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid

    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each
                          point contains the curvature ratio R for the
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = np.ndarray(DoG_pyramid.shape)
    for i in range(DoG_pyramid.shape[2]):
        DoG_image = DoG_pyramid[:, :, i]
        dxx = cv2.Sobel(DoG_image, ddepth=-1, dx=2, dy=0)
        dxy = cv2.Sobel(DoG_image, ddepth=-1, dx=1, dy=1)
        dyy = cv2.Sobel(DoG_image, ddepth=-1, dx=0, dy=2)
        principal_curvature[:, :, i] = (((dxx + dyy) ** 2) / (dxx * dyy - dxy * dxy))
    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    # filter based on 8 spatial and 2 level neighbors:
    #   x == 0 or x == imH or y == 0 or y == imW or
    #   principal_curvature[x][y][level] < th_r or
    #   abs(DoG_pyramid[x][y][level]) > th_contrast
    # If missing spatial neighbors, mark as invalid local extremum

    matches = (principal_curvature < th_r) & (abs(DoG_pyramid) > th_contrast)
    matches[0, :, :] = False
    matches[DoG_pyramid.shape[0] - 1, :, :] = False
    matches[:, 0, :] = False
    matches[:, DoG_pyramid.shape[1] - 1, :] = False

    locsDoG = np.argwhere(matches)
    locsDoG_final = []
    # Filter out based on neighbors
    for loc in locsDoG:
        x, y, level = loc
        loc_value = DoG_pyramid[x][y][level]
        # Get neighborhood
        neighborhood = DoG_pyramid[x - 1:x + 2, y - 1:y + 2, level].reshape((9))
        if level > 0:
            neighborhood = np.append(neighborhood, DoG_pyramid[x][y][level - 1])
        if level < DoG_pyramid.shape[2] - 1:
            neighborhood = np.append(neighborhood, DoG_pyramid[x][y][level + 1])
        if (neighborhood.max() == loc_value or neighborhood.min() == loc_value):
            # Swap coordinates
            locsDoG_final.append([loc[1], loc[0], loc[2]])
    locsDoG_final = np.asarray(locsDoG_final)
    return locsDoG_final


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    gauss_pyramid = createGaussianPyramid(im, sigma0=sigma0, k=k, levels=levels)
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid, levels=levels)
    principal_curvature = computePrincipalCurvature(DoG_pyramid)
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast=th_contrast, th_r=th_r)
    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    for loc in locsDoG:
        cv2.circle(im, (loc[0], loc[1]), 1, (0, 255, 0), -1)
    cv2.imshow('Keypoint Detector', im)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()
