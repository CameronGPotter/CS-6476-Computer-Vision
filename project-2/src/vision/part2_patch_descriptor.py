#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of key points
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    is_even = 1 if feature_width % 2 == 0 else 0
    D = feature_width * feature_width
    assert X.shape[0] == Y.shape[0]
    fvs = np.zeros((X.shape[0], D))
    for i, coordinates in enumerate(zip(Y, X)):
        row, col = coordinates[0], coordinates[1]
        idxs = [None] * 4
        idxs[0] = row - (feature_width // 2) + is_even
        idxs[1] = row + (feature_width // 2) + 1
        idxs[2] = col - (feature_width // 2) + is_even
        idxs[3] = col + (feature_width // 2) + 1
        window = image_bw[idxs[0]:idxs[1], idxs[2]:idxs[3]]
        normalized = window / np.linalg.norm(window)
        fvs[i, :] = normalized.flatten()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
