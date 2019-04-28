from __future__ import division
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

from BRIEF import briefLite, briefMatch


def main():
    im = cv2.imread('../data/model_chickenbroth.jpg')
    rows, cols = im.shape[:2]
    center = (rows//2, cols//2)
    scale = 1

    locs1, desc1 = briefLite(im)
    matches = []
    for angle in range(0, 360, 10):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        # Since rotated, columns become rows and rows become columns
        rotated_im = cv2.warpAffine(im, rotation_matrix, (cols, rows))
        locs2, desc2 = briefLite(rotated_im)
        brief_matches = briefMatch(desc1, desc2)
        num_matches = brief_matches.shape[0]
        matches.append(num_matches)

    y_pos = np.arange(len(matches)) * 10
    plt.bar(y_pos, matches, align='center', alpha=0.5)
    plt.xticks()
    plt.xlabel('Rotation angle (degrees)')
    plt.ylabel('Number of matches')
    plt.title('Number of matches vs. rotation angle for same image')
    plt.show()


if __name__ == '__main__':
    main()
