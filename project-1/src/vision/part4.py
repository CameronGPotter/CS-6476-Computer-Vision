#!/usr/bin/python3

import numpy as np


def my_conv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation. 
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the convolution in the frequency domain, and 
    - the result of the convolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        conv_result_freq: array of shape (m, n)
        conv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the convolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 for how to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    m, n = image.shape
    k, j = filter.shape
    m_extra = 0
    n_extra = 0
    if (m-k) % 2 == 1:
        m_extra = 1
    if (n-j) % 2 == 1:
        n_extra = 1
    image_freq = np.fft.fft2(image)
    filter_padded = np.pad(filter, pad_width=(((m-k)//2, m_extra + (m-k)//2), ((n-j)//2, n_extra + (n-j)//2)))
    filter_freq = np.fft.fft2(np.fft.ifftshift(filter_padded))
    conv_result_freq = filter_freq * image_freq
    conv_result = np.fft.ifft2(conv_result_freq)
    conv_result = np.real(conv_result)
    ### END OF STUDENT CODE ####
    ############################

    return image_freq, filter_freq, conv_result_freq, conv_result 


def my_deconv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation.
    
    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the deconvolution in the frequency domain, and 
    - the result of the deconvolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        deconv_result_freq: array of shape (m, n)
        deconv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the deconvolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 to see what this means and to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
    ### TODO: YOUR CODE HERE ###

    m, n = image.shape
    k, j = filter.shape
    m_extra = 0
    n_extra = 0
    if (m-k) % 2 == 1:
        m_extra = 1
    if (n-j) % 2 == 1:
        n_extra = 1
    image_freq = np.fft.fft2(image)
    filter_padded = np.pad(filter, pad_width=(((m-k)//2, m_extra + (m-k)//2), ((n-j)//2, n_extra + (n-j)//2)))
    filter_freq = np.fft.fft2(np.fft.ifftshift(filter_padded))
    deconv_result_freq = image_freq / filter_freq 
    deconv_result = np.fft.ifft2(deconv_result_freq)
    deconv_result = np.real(deconv_result)

    ### END OF STUDENT CODE ####
    ############################

    return image_freq, filter_freq, deconv_result_freq, deconv_result





