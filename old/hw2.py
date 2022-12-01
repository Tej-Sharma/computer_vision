# Filename: hw2.py
# Author: Tejas Sharma
# Date Created: 09/14/22
# Homework 2 - Create a Canny Edge Detector
# Note: Please press a key on each image after its displayed
# I have implemented it to display an image one by one
# It may take a few seconds for a kernel to be applied

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

    img = src.astype('float64')

    # Determine the width of the padding required. This
    # depends on the size of the kernel

    padding_width = kernel.shape[0] // 2

    # Retrieve the number of rows and columns of the original
    # image. You'll need this to create your padded image.

    rows = img.shape[0]
    cols = img.shape[1]

    # I'm using 0 padding, so I don't need to create a padded image
    # If the pixel is out of bounds, I just set the value to 0

    # Iterate through each pixel
    for (row, col), value in np.ndenumerate(src):
        sum = 0
        kernel_i, kernel_j = 0, 0
        # Apply the kernel
        for i in range(row - padding_width, row + padding_width + 1):
            for j in range(col - padding_width, col + padding_width + 1):
                # If it's out of bounds, use 0 padding by setting the val to 0
                if i < 0 or j < 0 or i >= rows or j >= cols:
                    img_val = 0
                else:
                    img_val = src[i][j]
                sum += img_val * kernel[kernel_i][kernel_j]
                kernel_j += 1
            kernel_j = 0
            kernel_i += 1
        img[row][col] = sum

    # Calculate the sum of your kernel
    kernel_sum = np.sum(kernel)

    # The sum of the kernel is greater than 1, then you
    # will need to divide your output array by the weight
    # of the kernel.
    if kernel_sum > 1:
        img = img / kernel_sum

    # return the output array
    return img

def clip(src):
    """
    Sets any value in the array over 255 to 255 and any value less
    than 0 to zero.
    """
    return np.clip(src, 0, 255)

def normalize(src):
    """
    Stretching the range of the image so that the minimal value is
    zero and the maximum value is 255
    """
    return (255 * (src - np.min(src)) / np.ptp(src))

def localMax(src):
    """
    Given an array, it finds the local maxima inside a window
    If the current pixel is not the maximum, make the pixel 0
    """
    window_size = 1
    output = src.copy().astype('float64')
    rows = output.shape[0]
    cols = output.shape[1]
    # Iterate for every pixel
    for row in range(rows):
        for col in range(cols):
            current_pixel = output[row][col]
            max_val = 0
            # Find the maximum value in the neighborhood (the window)
            for i in range(row - window_size, row + window_size + 1):
                for j in range(col - window_size, col + window_size + 1):
                    # Check for out of bounds
                    if not (i < 0 or i >= rows or j < 0 or j >= cols):
                        max_val = max(max_val, output[i][j])
            # If the current pixel is not the maximum, make it 0
            if current_pixel < max_val:
                output[row][col] = 0
    return output

def strongEdges(img):
    output = img.copy()
    rows = output.shape[0]
    cols = output.shape[1]
    strong_threshold = 135
    for row in range(rows):
        for col in range(cols):
            # Set any pixel greater than strong threshold to 255
            # Because this is a strong edge
            if output[row][col] >= strong_threshold:
                output[row][col] = 255
            else:
                # Otherwise set it to 0
                output[row][col] = 0
    return output

def weakEdges(img):
    output = img.copy()
    rows = output.shape[0]
    cols = output.shape[1]
    weak_threshold = 100
    for row in range(rows):
        for col in range(cols):
            # Eliminate any edges that are less than the weak threshold
            if output[row][col] < weak_threshold:
                output[row][col] = 0
            # However, do not set a weak edge to 255. 255 is reserved for strong edges
    return output


def finalDetection(strong, weak):
    """
    Promotes weak edges to strong edges if a weak edge
    is connected to a strong edge.
    """
    output = weak.copy()
    rows = output.shape[0]
    cols = output.shape[1]

    window_size = 1

    for row in range(rows):
        for col in range(cols):
            connected_to_strong_edge = False
            # Check if this pixel is connected to a strong edge
            # Note: we count diagonal pixels as connected
            # That's why a window is being used
            for i in range(row - window_size, row + window_size + 1):
                for j in range(col - window_size, col + window_size + 1):
                    if not (i < 0 or i >= rows or j < 0 or j >= cols):
                        if strong[i][j] == 255:
                            connected_to_strong_edge = True

            # If this is a weak edge ( > 0), promote it to a strong edge
            if output[row][col] > 0 and connected_to_strong_edge:
                output[row][col] = 255
    return output

def display_image(window_name, img_arr):
    cv2.imshow(window_name, img_arr)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Select the image to open
    filename = askopenfilename()

    # Create a threshold for weak and strong edges. This
    # value should range between zero and 255. I have 
    # selected some arbitrary values. You are welcome to 
    # change them, as these are dependent on the input image
    strongthreshold = 150
    weakthreshold = 100

    # Open the image in grayscale
    src = cv2.imread(filename,0);
    cv2.imshow("Input", src)

    # Begin by smoothing the image using a Gaussian blur.
    # You can use your convolution function to do so or call
    # the OpenCV function.

    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype='float64')
    dst = convolution(src, kernel)
    cv2.imshow("Smooth Image", dst.astype('uint8'))

    cv2.waitKey(0)

    # Next, find the gradients in the x and y directions. You
    # will have two separate output arrays. Use the Sobel
    # kernels to perform two convolutions, one for horizontal
    # and the other for vertical gradients.

    vertical_lines = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float64')
    vertical_separations_img = convolution(src, vertical_lines)

    horizontal_lines = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype='float64')
    horizontal_separations_img = convolution(src, horizontal_lines)
    
    # Display the output of the two convolutions using imshow
    # Keep in mind that your resulting array with have values
    # that are between -4*255 and 4*255. Pixels with no change
    # in horiz or vertical gradient should appear as gray (127)

    cv2.imshow("Vertical Seprations Image", normalize(vertical_separations_img).astype('uint8'))
    cv2.waitKey(0)

    cv2.imshow("Horizontal Seprations Image", normalize(horizontal_separations_img).astype('uint8'))
    cv2.waitKey(0)


    # Compute the gradient matrix by taking the square root of
    # the sum of the squares of the gradient matrices.

    combined_gradient_matrix = np.sqrt(vertical_separations_img ** 2 + horizontal_separations_img ** 2)
    cv2.imshow("Gradient", combined_gradient_matrix.astype('uint8'))
    cv2.waitKey(0)

    # Compute non-maximum suppression for the single gradient
    # array.

    non_max_suppressed = localMax(normalize(combined_gradient_matrix))
    cv2.imshow("Local Max (Non-Maximum Suppressed)", non_max_suppressed.astype('uint8'))
    cv2.waitKey(0)


    # Create two arrays for strong and weak edges. In the strong
    # edge image, any values that are above the strong threshold
    # are considered strong. Weak edges are edges that are above
    # the weak threshold but below the strong edge threshold

    strong_edges_image = strongEdges(non_max_suppressed.astype('uint8')).astype('uint8')
    weak_edges_image = weakEdges(non_max_suppressed.astype('uint8')).astype('uint8')

    cv2.imshow("Strong Edges", strong_edges_image)
    cv2.waitKey(0)
    cv2.imshow("Weak Edges", weak_edges_image)
    cv2.waitKey(0)


    # Final detection. Any weak edge that touches a strong edge is
    # promoted to strong edge. that are Combine Weak and Strong Edges

    final_image = finalDetection(strong_edges_image, weak_edges_image)
    cv2.imshow("Final Detection", final_image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    """
    cv2.imshow("Gradient", src)
    cv2.imshow("Local Max", src)
    cv2.imshow("Final Detection", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
