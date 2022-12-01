# Filename:
# Author:
# Date Created:
# Image Morphology

import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

def dilate(src, mask):
    """
    The value of the output pixel is the maximum of all values
    in the neighborhood. Neighborhood is defined by the mask.
    """
    rows = src.shape[0]
    cols = src.shape[1]

    img = np.copy(src)
    non_zero_vals = np.nonzero(img)

    for i in range(len(non_zero_vals[0])):
        row = non_zero_vals[0][i]
        col = non_zero_vals[1][i]

        max_val = float('-inf')
        for i in range(row - mask // 2, row + mask // 2 + 1):
            for j in range(col - mask // 2, col + mask // 2 + 1):
                # Do not include current pixel or out of bounds
                if i > 0 and i < rows and j > 0 and j < cols:
                    if i == row and j == col:
                        pass
                    else:
                        max_val = max(img[i][j], max_val)
        img[row][col] = max_val

    return img

def erode(src, mask):
    """
    The value of the output pixel is the minimum of all values
    in the neighborhood. Neighborhood is defined by the mask.
    """
    rows = src.shape[0]
    cols = src.shape[1]

    img = np.copy(src)

    non_zero_vals = np.nonzero(img)

    for i in range(len(non_zero_vals[0])):
        row = non_zero_vals[0][i]
        col = non_zero_vals[1][i]

        min_val = float('inf')
        for i in range(row - mask // 2, row + mask // 2 + 1):
            for j in range(col - mask // 2, col + mask // 2 + 1):
                # Do not include current pixel or out of bounds
                if i > 0 and i < rows and j > 0 and j < cols:
                    if i == row and j == col:
                        pass
                    else:
                        min_val = min(img[i][j], min_val)
        img[row][col] = min_val

    return img

def morph_open(src, mask):
    """
    Erosion followed by a dilation.
    """

    return src

def morph_close(src, mask):
    """
    Dilation followed by an erosion.
    """

    return src

def main():
    """
    Performs four image morphological operators
    (dilation, erosion, opening, closing) on an
    input image.
    """
    # Part 1 - Read in the image and convert the image
    # to grayscale
    filename = askopenfilename()
    src = cv2.imread(filename, 0)

    # Display the grayscale image
    cv2.imshow('Grayscale', src)

    # Create a threshold for converting the grayscale
    # image into a binary image
    threshold = 150

    # Using the threshold, create a binary image.
    # Anything above the threshold is set to 255. Anything
    # below is set to zero.
    for (row, col), value in np.ndenumerate(src):
        if value < threshold:
            src[row][col] = 0
        else:
            src[row][col] = 255

    # Display the thresholded image
    cv2.imshow('Threshold Image', src)

    # Implement the erosion, dilation, open, and close function
    # Display all four resulting images.

    dilated_image = dilate(src, 5)
    cv2.imshow('Dilated Image', dilated_image)

    eroded_image = erode(src, 5)
    cv2.imshow('Eroded Image', eroded_image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
