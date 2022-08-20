import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################
    is_windows = os.name=='nt'
    img_list = np.array([])
    for fld in ['/train/', '/test/']:
        path = (dir_name + fld)
        if is_windows:
            path = path.replace('\\', '/')
            # path = path.replace('/', '\\')
        for subdir, _, files in os.walk(path):
            for file in files:
                filepath = os.path.join(subdir, file)
                if is_windows:
                    filepath = filepath.replace('\\', '/')
                    # filepath = filepath.replace('/', '\\')
                img = np.array(Image.open(filepath).convert('L')).astype(np.float32)
                img /= 255
                img_list = np.append(img_list, img)
    
    mean = np.mean(img_list)
    std = np.std(img_list, ddof=1)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
