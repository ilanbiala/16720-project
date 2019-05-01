import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    #im1 = cv2.imread('../photos/good1-cropped-extra-no-bg.png')
    #im2 = cv2.imread('../photos/good2-cropped-extra-no-bg.png')
    im1 = cv2.imread('../Palace/00136.jpg')
    im2 = cv2.imread('../Palace/00137.jpg')

    height, width, _ = im1.shape
    im1 = cv2.resize(im1, (int(height/4), int(width/4)))
    im2 = cv2.resize(im2, (int(height/4), int(width/4)))
 
    surf = cv2.xfeatures2d.SURF_create(400)
    locs1, desc1 = surf.detectAndCompute(im1, None)
    locs2, desc2 = surf.detectAndCompute(im2, None)
    print(desc1)
    #matches = briefMatch(desc1, desc2)
    im = cv2.drawKeypoints(im1, locs1, None, (255, 0, 0), 4)
    plt.imshow(im)
    #plt.show()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(im1,locs1,im2,locs2,matches,None,**draw_params)
    #plt.imshow(img3,),plt.show()
    locs1 = [loc.pt for loc in locs1]
    locs2 = [loc.pt for loc in locs2]
    #for i, (m n) in enumerate(matches):
        #matches
    #np.savez('../data/temple_image', locs1=locs1, locs2=locs2, matches=matches, im1=im1, im2=im2)
