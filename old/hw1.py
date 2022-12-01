# Name: Tejas Sharma
# Date: 09/01/2022
# Assignment: Homework 1
#
# Villanova CSC5930/9010 Computer Vision
#
# Image Manipulation - rotations, transformations, etc.
# Took 2 hours to complete

import cv2
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

from tkinter.filedialog import askopenfilename


def split_image(src):
    """
    Given a three channel image, this function splits the image
    into blue, green, and red and returns each channel separately
    """
    blue = src[:, :, 0]
    green = src[:, :, 1]
    red = src[:, :, 2]
    return (blue, green, red)

def clip(src):
    """
    Sets any value in the array over 255 to 255 and any value less
    than 0 to zero.
    """
    return np.clip(src, 0, 255)

def brighten(src, val):
    """
    Brightens the image by uniformly increasing each pixel intensity
    by the given value
    """
    img = src.astype('int16')
    result = img + val
    result = clip(result)
    result = result.astype('uint8')
    return result


def darken(src, val):
    """
    Darkens the image by uniformly increasing each pixel intensity
    by the given value
    """
    src = src.astype('int16')
    result = src - val
    result = clip(result)
    result = result.astype('uint8')
    return result


def normalize(src):
    """
    Stretching the range of the image so that the minimal value is
    zero and the maximum value is 255
    """
    # Normalize to 0 to 1 by following the normalization formula
    return clip(src / np.sqrt(np.sum(src**2)))

def pad(src, width, val=0):
    """
    Pads the image with a constant value given the specified width
    """
    # I found this function on the opencv documentation that adds a padding
    padded_image = cv2.copyMakeBorder(src, width, width, width, width, cv2.BORDER_CONSTANT, value=val)
    return padded_image

def clockwise(src):
    """
    Rotates the image clockwise 90 degrees
    """
    return np.rot90(src, 3)


def cclockwise(src):
    """
    Rotates the image counter-clockwise 90 degrees
    """
    return np.rot90(src, 1)


def quadrants(src):
    """
    Splits the image into four regions and returns them
    in the following order:
    top-left, top-right, bottom-left, bottom-right
    """
    height, width, channels = src.shape

    images = []
    for i in range(2):
        for j in range(2):
            x = int(width / 2 * i)
            y = int(height / 2 * j)
            h = int(height / 2)
            w = int(width / 2)
            img = src[y : y + h, x:x + w]
            images.append(img)
    return images


def downscale(src):
    """
    Returns an array half the size by removing every other element
    from the rows and columns
    """
    # Remove every other element to create a smaller image
    # (alternate: is to take average around all pixels and make it that)

    height, width, channels = src.shape
    result_image = np.empty((height // 2, width // 2, channels), dtype='uint8')

    # Use a step of 2 to skip over elements
    y, x = 0, 0
    for i in range(0, height, 2):
        x = 0
        for j in range(0, width, 2):
            for k in range(0, 3):
                result_image[y][x] = src[i][j]
            x += 1
        y += 1
    return result_image


def upscale(src):
    """
    Returns an array twice the input size by duplicating neighboring
    values in the rows and columns
    NOTE: for an upscaled image, if it exceeds screen size, opencv downscales it
    """
    height, width, channels = src.shape
    result_image = np.zeros((height * 2, width * 2, channels), dtype='uint8')

    # Instead of using a step of 2 for src, use a step of 2 for destination
    y, x = 0, 0
    for i in range(0, height):
        x = 0
        for j in range(0, width):
            for k in range(0, 3):
                result_image[y][x] = src[i][j]
                result_image[y][x+1] = src[i][j]
                result_image[y+1][x+1] = src[i][j]
                result_image[y+1][x] = src[i][j]
            x += 2
        y += 2

    return result_image


def grayscale(src, b, g, r):
    """
    Converts a 3 dimensional array (color image) to one-dimensional array
    (grayscale image) using the weights given for b, g, and r
    b, g, and r are floating point values that should sum to 1.0
    """
    height, width, channels = src.shape
    result_image = np.zeros((height, width, channels), dtype='uint8')
    for i in range(height):
        for j in range(width):
            result_image[i, j] = sum(src[i, j]) * sum(np.array([b, g, r]) / 3)
    return result_image

def display_image(window_name, img_arr):
    cv2.imshow(window_name, img_arr)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def main():
    """
    This program performs an operation on the image, waits one second, closes the window,
    and then moves onto the next operation.
    Note: UpScale and GrayScale take a few seconds of loading depending on the image size
    """
    # Select the image that you will be working with for this assignment.
    # You should select a 3-channel color image.
    filename = askopenfilename()

    # Read the image as-is (don't resize, convert to grayscale, etc.)
    # Save the image to a variable called src. This will be the source image.
    src = cv2.imread(filename)

    # Implement the function split_image above to split the image into
    # blue, red, and green channels and display all three images
    bgr_split = split_image(src)

    display_image('blue_image', bgr_split[0])
    display_image('green_image', bgr_split[1])
    display_image('red_image', bgr_split[2])

    # Implement the function brighten image and display the resulting
    # image when increasing the images intensity by 20
    brightened_image = brighten(src, 20)
    display_image('brightened_image', brightened_image)

    # Implement the function darken image and display the resulting
    # image when increasing the images intensity by 20
    darkened_image = darken(src, 20)
    display_image('darkened_image', darkened_image)

    # Implement the function normalize and display the resulting
    # image. Call this function 3 times and show each image channel from
    # your input image

    normalized_image = normalize(src)
    display_image('normalized_image', normalized_image)

    # Implement the function pad to place a uniform constant value
    # around the border of the image. Display the resulting image.

    padded_image = pad(src, 50, 0)
    display_image('padded_image', padded_image)

    # Implement the function clockwise to rotate your image 90 degrees.
    # Display the resulting image.

    rotated_image = clockwise(src)
    display_image('clockwise_90_image', rotated_image)


    # Implement the function cclockwise to rotate your image 90 degrees
    # counterclockwise. Display the resulting image.

    rotated_image = cclockwise(src)
    display_image('counter_clockwise_90_image', rotated_image)

    # Implement the function quadrants which divides the image equally
    # into four components. Display the resulting four images.

    four_images = quadrants(src)
    for i in range(len(four_images)):
        display_image('quadrant image ' + str(i), four_images[i])

    # Implement the function downscale that shrinks the image by a factor
    # of two in the horizontal and vertical direction and display the
    # resulting image.
    downscaled_img = downscale(src)
    display_image('downscaled image', downscaled_img)

    # Implement the function upscale that enlarges the image by a factor
    # of two in the horizontal and vertical direction and display the
    # resulting image.

    upscaled_img = upscale(src)
    display_image('upscaled image', upscaled_img)

    # Implement the function grayscale that takes in a 3-channel image
    # and creates a grayscale image based upon the scaled values given.

    grayscaled_image = grayscale(src, 0.35, 0.35, 0.3)
    display_image('grayscale image', grayscaled_image)

    # You should display and close the image after each section.


if __name__ == "__main__":
    main()

