import cv2
import numpy as np

def calc_mean_gray(image):
    """
    Calculate the mean gray value of an image.

    Parameters:
    image (numpy.ndarray): Input image in BGR format.

    Returns:
    float: Mean gray value of the image.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the mean gray value
    mean_gray = np.mean(gray_image)
    median_gray = np.median(gray_image)
    
    return mean_gray, median_gray

def calc_mean_value(image):
    """
    Calculate the mean value of the V channel in HSV color space.

    Parameters:
    image (numpy.ndarray): Input image in BGR format.

    Returns:
    float: Mean value of the V channel.
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract the V channel
    v_channel = hsv_image[:, :, 2]
    
    # Calculate the mean value of the V channel
    mean_v = np.mean(v_channel)
    median_v = np.median(v_channel)

    return mean_v, median_v

