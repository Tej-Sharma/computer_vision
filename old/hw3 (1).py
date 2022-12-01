# Filename: hw3.py
# Author: Tejas Sharma
# Date Created: 09/15/22
# Homework 3: Object Segmentation
#
# This assignment counts the number of shapes in the images

import cv2
import sys
import numpy as np
from tkinter.filedialog import askopenfilename
from collections import defaultdict

np.set_printoptions(threshold=sys.maxsize)


def conncomponents(src):
    """
    A two pass approach for finding connected components. Assigns a
    foreground pixel a new label if its left and upper neighbors
    are background pixels. Otherwise, the lower of the two values are
    assigned to it. The second pass recifies the pixels that have
    corresponding values.

    Input should be a binary image where black pixels are background
    and foreground pixels (objects) are white.
    """

    rows = src.shape[0]
    cols = src.shape[1]

    # This will store the final image and assign labels to each pixel
    labels = np.zeros((rows, cols), dtype=np.int64)

    # Map a label to the equivalent label (i.e. a label of 3 and 1 are the same image)
    equivalent_labels = defaultdict(lambda: 99999)

    # Make first pass to assign labels to each pixel
    # We check if it's a foreground pixel
    # Then we check if its left or uppper has a label
    # If there's two conflicting neighbors, we assign the minimum one
    label_to_assign = 1

    for i in range(rows):
        for j in range(cols):
            curr_possible_labels = []
            if src[i][j] == 255:
                # Check if its neighbors have a label
                if j - 1 >= 0 and labels[i][j - 1] != 0:
                    curr_possible_labels.append(labels[i][j - 1])
                if i - 1 >= 0 and labels[i - 1][j] != 0:
                    curr_possible_labels.append(labels[i - 1][j])
                # No neighbors with a label
                if len(curr_possible_labels) == 0:
                    labels[i][j] = label_to_assign
                    label_to_assign += 1
                else:
                    min_nearby_label = min(curr_possible_labels)
                    labels[i][j] = min_nearby_label
                    for label in curr_possible_labels:
                        if equivalent_labels[label] > min_nearby_label and label != min_nearby_label:
                            equivalent_labels[label] = min_nearby_label

    # Combine all the equivalent labels together
    # so we only get the unique, root labels
    # This is kind of like the Union Find algorithm
    for label in equivalent_labels:
        matching_label = equivalent_labels[label]
        while matching_label in equivalent_labels and equivalent_labels[matching_label] < 99999:
            matching_label = equivalent_labels[matching_label]
        equivalent_labels[label] = matching_label
    
    # Aggregate only the unique labels
    # So find the number of blobs we have (# of root labels)
    unique_labels = []
    for label in equivalent_labels:
        if equivalent_labels[label] == 99999:
            if label not in unique_labels:
                unique_labels.append(label)
        elif equivalent_labels[label] not in unique_labels:
            unique_labels.append(equivalent_labels[label])

    # Print the number of images found
    print("There are {} unique objects in the image!".format(len(unique_labels)))
    
    # Now assign each label a scaled color based on the index
    # so (0, 1, 2...) * scale
    scale = max(1, 255 // len(unique_labels))

    for i in range(rows):
        for j in range(cols):
            label = labels[i][j]
            if label != 0:
                # If it's not 0, there's an image here
                if equivalent_labels[label] != 99999:
                    labels[i][j] = (unique_labels.index(equivalent_labels[label]) + 1) * scale
                else:
                    try:
                        labels[i][j] = (unique_labels.index(label) + 1) * scale
                    except:
                        labels[i][j] = 255

    # This is the final image
    return labels.astype(np.uint8)

if __name__ == "__main__":

    filename = askopenfilename()

    # Read in the image and convert the image to binary.
    src = cv2.imread(filename,0)
    cv2.imshow("Input", src)

    # Using the threshold, create a binary image.
    # Anything above the threshold is set to 255. Anything
    # below is set to zero.
    for (row, col), value in np.ndenumerate(src):
        if value >= 0 and value <= 200:
            src[row][col] = 255
        else:
            src[row][col] = 0

    cv2.imshow("Thresholded Image", src)

    # Run the connected components algorithm.

    # Print out the number of unique values from the connected
    # components algorithm. This is the number of shapes, including
    # the background.

    # Scale the shape labels (1,2,3,4...) between 0 and 255 so
    # that these values can been seen in a grayscale image.

    blobs = conncomponents(src)
    cv2.imshow("Blobs Image", blobs)

    # Bonus: Color the shapes (randomly)

    # Use this open cv function that applies a color map based on the labels
    blobs_colored = cv2.applyColorMap(blobs, cv2.COLORMAP_JET)
    cv2.imshow("Colored Blobs", blobs_colored)

    cv2.waitKey(0)
    cv2.destroyAllWindows()