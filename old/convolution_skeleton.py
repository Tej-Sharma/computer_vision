# Filename: convolution.py
# Author: Jason Grant, Villanova University
# Date Created: 8 Sept 2022
# 
# Convolution

import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

def convolution(src, kernel):
    """
    Performs the convolution of an input image with 
    the given kernel.
    """
    
    # Create a new output array that is the same size as
    # your input array. You'll have values larger than
    # 255 so you'll want to use a data type that can hold
    # larger numbers.

    img = src.astype('int32')

    # Determine the width of the padding required. This
    # depends on the size of the kernel

    padding_width = kernel.shape[0] // 2

    # Retrieve the number of rows and columns of the original
    # image. You'll need this to create your padded image.

    rows = img.shape[0]
    cols = img.shape[1]

    # Create your padded image. You can use either the OpenCV
    # function, the NumPy implementation, or your own
    # implementation to pad the image. You are also free to
    # choose the type of padding. This could be a constant
    # value, a stretched border, mirrored border, etc.

    # Add 0 padding
    padded_image = cv2.copyMakeBorder(src, padding_width, padding_width, padding_width, padding_width, cv2.BORDER_CONSTANT, value=0)


    # Retrieve the number of rows and columns of the kernel
    # and store those in a variable

    kernel_rows = kernel.shape[0]
    kernel_cols = kernel.shape[1]

    # Iterate through each pixel of the image. Multiply the
    # values in the kernel by the overlapping pixel values 
    # and sum the values. Place the resulting value in the
    # output array

    for (y, x), value in np.ndenumerate(src):
        sum = 0
        kernel_i, kernel_j = 0, 0
        for i in range(y-padding_width, y+padding_width+1):
            for j in range(x-padding_width, x+padding_width+1):
                img_val = None
                if i < 0 or j < 0 or i >= rows or j >= cols:
                    img_val = 0
                else:
                    img_val = src[i][j]
                sum += img_val * kernel[kernel_i][kernel_j]
                kernel_j += 1
            kernel_j = 0
            kernel_i += 1
        img[y][x] = sum

    # Calculate the sum of your kernel
    kernel_sum = np.sum(kernel)

    # The sum of the kernel is greater than 1, then you
    # will need to divide your output array by the weight
    # of the kernel.

    if kernel_sum > 1:
        img = img / kernel_sum

    # return the output array
    return img

def main():
    """
    Reads in an image, performs the convolution based upon
    the kernel, and displays the output image.
    """

    # Select a file
    filename = askopenfilename()

    # Open the file and convert it to grayscale.
    src = cv2.imread(filename, 0)

    # Create a kernel. You can modify the identity kernel
    # given to you below.
    kernel = np.array([[1,1,1],[0,1,0],[1,0,0]])

    # Perform the convolution
    dst = convolution(src, kernel)
    
    # Display the input and output image
    cv2.imshow('Input', src)
    cv2.imshow('Output', dst.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
