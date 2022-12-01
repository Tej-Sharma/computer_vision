# Filename: hw4.py
# Author: Tejas Sharma
# Date Created: 09/22
#
# Hough Transformation for finding lines in an image

import cv2
import numpy as np
from tkinter.filedialog import askopenfilename
import time

def convolution(src, kernel):
    """
    Convolves an input image with the given kernel.
    """

    # Copy your code from class here. Please do not import your
    # convolution file and call that function from here. It
    # will cause issues when grading.

    dst = np.zeros(src.shape,np.int32)

    rows = src.shape[0]
    cols = src.shape[1]
    klen = kernel.shape[0]
    padwid = klen//2

    padimg = np.pad(src, padwid*2, mode = 'reflect')

    for row in range(rows):
        for col in range(cols):
            
            dst[row,col] = np.sum(np.multiply(padimg[row:row+klen,col:col+klen],kernel))

    ksum = np.sum(kernel)
    return dst / ksum if ksum > 1 else dst 

def calcdirgrad(horiz, vert, threshold=5):
    """
    Calculates the approximate angle from the horizontal
    and vertical gradients.
    """

    # Arctan vertical with horizontal
    output = np.arctan2(vert, horiz)
    
    # Convert output to degrees
    output = output * 180 / np.pi

    return output

def incrementAccumulator(src, H):
    """
    Given an input image, this function increments the
    accumulator for each point in the source (edge) image
    """

    # Setting every other pixel to zero to reduce the amount
    # of computation
    tmp = src.copy()
    tmp[::2,::2] = 0

    # Gets all of the nonzero values of the edge image
    rows = tmp.shape[0]
    cols = tmp.shape[1]

    # Each point in the image could lie on a line
    # in any direction. So check every direction.
    # a better approximation would be to use information
    # the gradient to limit the computations. The gradient
    # information would give us an approximate angle, instead
    # of checking 180 degrees. Then, increment the accumulator
    # for that angle.
    for i in range(rows):
        for j in range(cols):
            if tmp[i,j] > 0:
                for theta in range(180):
                    rho = int(i * np.cos(np.pi * theta/180) + j * np.sin(np.pi * theta/180))
                    H[theta, rho] += 1

def scaleArray(src, nmin=0, nmax=255):
    """
    Given an input array, this function scales they
    array between the minimum and maximum value, and
    returns an np.uint8 array, so that the array can be displayed
    as an image.
    """

    src = src.astype(np.float32)

    min = np.amin(src)
    max = np.amax(src)
    range = max - min

    src = src - min
    src = src / range
    src = src * 255

    return src.astype(np.uint8)

def localMax(src):
    """
    Given an array, it finds the local maxima in the
    x direction and y directions.
    """

    # Copy your code from class here. Please do not import your
    # localMax file and call that function from here. It
    # will cause issues when grading.
    
    rows = src.shape[0]
    cols = src.shape[1]
    local_max = np.zeros(src.shape,np.int32)
    window_size = 1

    for row in range(rows):
        for col in range(cols):
            current_pixel = src[row][col]
            max_val = 0
            # Find the maximum value in the neighborhood (the window)
            for i in range(row - window_size, row + window_size + 1):
                for j in range(col - window_size, col + window_size + 1):
                    # Check for out of bounds
                    if not (i < 0 or i >= rows or j < 0 or j >= cols):
                        max_val = max(max_val, src[i][j])

            # If the current pixel is not the maximum, make it 0
            if current_pixel >= max_val:
                local_max[row][col] = current_pixel
                print(local_max[row][col])

    return local_max


def drawpolarlines(src, local_max_array, maxrho):
    """
    Draws the lines onton the output image
    """

    houghlines = src.copy()
    
    # Gets all of the nonzero values of the accumulator
    polarlines = np.transpose(np.nonzero(local_max_array))
    
    print("Drawing lines on source image")
    for i in range(polarlines.shape[0]):

        theta, rho = polarlines[i,:]
        a = np.cos(np.pi * theta/180)
        b = np.sin(np.pi * theta/180)

        x0 = int(a * (rho-maxrho))
        y0 = int(b * (rho-maxrho))
        x1 = int(x0 - b * maxrho)
        y1 = int(y0 + a * maxrho)
        x2 = int(x0 + b * maxrho)
        y2 = int(y0 - a * maxrho)

        cv2.line(src,(x1,y1),(x2,y2), (0,255,0),2)

        return houghlines

def main():
    """
    Finds lines in an image using the Hough Transformation.
    """
    
    # Reads in the file and coverts it to grayscale. This
    # program expects the input file to be a
    filename = askopenfilename()
    src = cv2.imread(filename)
    gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    time.sleep(1)
    
    # Create all of the windows that will be needed for this assignment.
    cv2.namedWindow("Source",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edge Image",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Hough Lines",cv2.WINDOW_NORMAL)

    cv2.imshow("Source", src)
    cv2.waitKey(10)

    # Create a Canny edge image from the input image
    print("Creating Edge Image")
    gray = cv2.Canny(gray,100,150)
    cv2.imshow("Edge Image", gray)
    cv2.waitKey(10)

    # Calculate the horizontal and vertical gradients and find
    # the approximate gradient angle each pixel.

    vertical_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical_separations_img = convolution(gray, vertical_sobel)
    horizontal_sobel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    horizontal_separations_img = convolution(gray, horizontal_sobel)

    gradients = calcdirgrad(horizontal_sobel, vertical_sobel)

    # Calculate the maximum length a line could be in the image
    maxrho = np.ceil(max(gray.shape[0],gray.shape[1]) * 1.414)

    # Accumulator Matrix
    H = np.zeros((180,int(2*maxrho)),np.uint32)

    # For each point, increment the accumulator.
    incrementAccumulator(gray, H)

    # Finds the local maximas in the accumulator (call localMax)
    local_max_array = localMax(H)


    # draw Lines
    houghlines = drawpolarlines(src, local_max_array, maxrho)


    cv2.imshow("Source", gray)
    cv2.imshow("Hough Lines",houghlines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
