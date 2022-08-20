"""Fundamental matrix utilities."""

from email.errors import MultipartConversionError
import numpy as np
from torch import mean


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################s
    mean_points = np.sum(points, axis=0) / points.shape[0]
    means = np.identity(3)
    means[0:2, 2] = (-1 * mean_points).T

    std1 = np.array(points, dtype=float)
    std1[:, 0] = std1[:, 0] - mean_points[0]
    std1[:, 1] = std1[:, 1] - mean_points[1]
    scale_factor = np.diag(np.hstack((np.reciprocal(np.std(std1, axis=0)), [1])))

    T = np.dot(scale_factor, means)
    
    homogenous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_normalized = np.zeros(points.shape)
    for i, row in enumerate(homogenous):
        mult = np.dot(T, row)
        points_normalized[i] = mult[0:2]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = T_b.T @ F_norm @ T_a

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    points_a, T_a = normalize_points(points_a)
    points_b, T_b = normalize_points(points_b)

    homogenous_a = np.hstack((points_a, np.ones((points_a.shape[0], 1))))
    homogenous_b = np.hstack((points_b, np.ones((points_b.shape[0], 1))))

    A = np.ones((points_a.shape[0], 8))
    A[:, 0] = np.multiply(homogenous_b[:, 0], homogenous_a[:, 0])
    A[:, 1] = np.multiply(homogenous_b[:, 0], homogenous_a[:, 1])
    A[:, 2] = np.multiply(homogenous_b[:, 0], homogenous_a[:, 2])
    A[:, 3] = np.multiply(homogenous_b[:, 1], homogenous_a[:, 0])
    A[:, 4] = np.multiply(homogenous_b[:, 1], homogenous_a[:, 1])
    A[:, 5] = np.multiply(homogenous_b[:, 1], homogenous_a[:, 2])
    A[:, 6] = homogenous_a[:, 0]
    A[:, 7] = homogenous_a[:, 1]


    f = np.linalg.lstsq(A, -1*np.ones((A.shape[0], 1)), rcond=None)[0]
    f = np.vstack((f, [1]))
    F = np.reshape(f, (3, 3))

    U, D, V = np.linalg.svd(F)

    i = np.where(D == np.min(D[np.nonzero(D)]))
    D[i] = 0

    F = U @ np.diag(D) @ V

    F = unnormalize_F(F, T_a, T_b)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
