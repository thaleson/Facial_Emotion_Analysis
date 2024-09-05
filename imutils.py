# Imports
import numpy as np
import cv2

def translate(image, x, y):
    """
    Translates an image by shifting it along the x and y axes.

    Parameters:
    - image (numpy.ndarray): The image to be translated.
    - x (int): The number of pixels to shift the image horizontally.
    - y (int): The number of pixels to shift the image vertically.

    Returns:
    - numpy.ndarray: The translated image.
    """
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Return the translated image
    return shifted

def rotate(image, angle, center=None, scale=1.0):
    """
    Rotates an image by a specified angle around a given center point.

    Parameters:
    - image (numpy.ndarray): The image to be rotated.
    - angle (float): The angle in degrees to rotate the image.
    - center (tuple, optional): The (x, y) coordinates of the center of rotation. If None, the center of the image is used.
    - scale (float, optional): The scale factor for the rotation. Default is 1.0 (no scaling).

    Returns:
    - numpy.ndarray: The rotated image.
    """
    # Get the dimensions of the image
    (h, w) = image.shape[:2]

    # If center is None, initialize it to the center of the image
    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
    return rotated

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resizes an image to the specified width and/or height while maintaining aspect ratio.

    Parameters:
    - image (numpy.ndarray): The image to be resized.
    - width (int, optional): The desired width of the resized image. If None, the height is used.
    - height (int, optional): The desired height of the resized image. If None, the width is used.
    - inter (int, optional): The interpolation method used for resizing. Default is cv2.INTER_AREA.

    Returns:
    - numpy.ndarray: The resized image.
    """
    # Initialize the dimensions of the resized image and get the image size
    dim = None
    (h, w) = image.shape[:2]

    # If both width and height are None, return the original image
    if width is None and height is None:
        return image

    # Check if width is None
    if width is None:
        # Calculate the ratio of height and build the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # Otherwise, height is None
    else:
        # Calculate the ratio of width and build the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # Return the resized image
    return resized
