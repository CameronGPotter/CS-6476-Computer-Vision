import numpy as np
import cv2 as cv
from vision.part3_ransac import ransac_fundamental_matrix
from vision.utils import load_image, save_image, get_matches
import matplotlib.pyplot as plt

def panorama_stitch(imageA, imageB):
    """
    ImageA and ImageB will be an image pair that you choose to stitch together
    to create your panorama. This can be your own image pair that you believe
    will give you the best stitched panorama. Feel free to play around with 
    different image pairs as a fun exercise!
    
    Please note that you can use your fundamental matrix estimation from part3
    (imported for you above) to compute the homography matrix that you will 
    need to stitch the panorama.
    
    Feel free to reuse your interest point pipeline from project 2, or you may
    choose to use any existing interest point/feature matching functions from
    OpenCV. You may NOT use any pre-existing warping function though.

    Args:
        imageA: first image that we are looking at (from camera view 1) [A x B]
        imageB: second image that we are looking at (from camera view 2) [M x N]

    Returns:
        panorama: stitch of image 1 and image 2 using warp. Ideal dimensions
            are either:
            1. A or M x (B + N)
                    OR
            2. (A + M) x B or N)
    """
    panorama = None

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    imageA_cv = cv.imread(imageA)
    imageA_bw = cv.cvtColor(imageA_cv, cv.COLOR_BGR2GRAY)

    imageB_cv = cv.imread(imageB)
    imageB_bw = cv.cvtColor(imageB_cv, cv.COLOR_BGR2GRAY)

    final_W = imageB_cv.shape[1] + imageA_cv.shape[1]
    final_H = imageB_cv.shape[0]

    cv_SIFT = cv.xfeatures2d.SIFT_create()

    kpA, desA = cv_SIFT.detectAndCompute(imageA_bw, None)
    kpB, desB = cv_SIFT.detectAndCompute(imageB_bw, None)

    init_matches = cv.BFMatcher().knnMatch(desA, desB, k=2) 

    useful_matches = []
    for match in init_matches:
        if match[1].distance * 0.5 > match[0].distance:         
            useful_matches.append(match)
            
    matches = np.asarray(useful_matches)

    source_list = []
    destination_list = []
    for match in matches[:,0]:
        source_list.append(kpA[match.queryIdx].pt)
        destination_list.append(kpB[match.trainIdx].pt)
    
    np_source = np.float32(source_list).reshape(-1, 1, 2)
    np_destination = np.float32(destination_list).reshape(-1, 1, 2)

    homography_matrix = cv.findHomography(np_source, np_destination, cv.RANSAC, 5.0)[0]

    index_points = np.mgrid[0:final_W, 0:final_H].reshape(2, -1).T
    padded = np.pad(index_points, [(0, 0), (0, 1)], constant_values=1)
    points = np.dot(np.linalg.inv(homography_matrix), padded.T).T
    
    mapped_points = (points / points[:, 2].reshape(-1, 1))[:, 0:2].reshape(final_W, final_H, 2).astype(np.float32)

    panorama = cv.remap(imageA_cv, mapped_points, None, cv.INTER_CUBIC).transpose(1, 0, 2)

    panorama[0:imageB_cv.shape[0], 0:imageB_cv.shape[1]] = imageB_cv

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return panorama
