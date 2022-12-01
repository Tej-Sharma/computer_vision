# Filename: Video.py
# Author: Tejas Sharma
# Date Created: 11/02/2022
# I'm using the 'x' key to start selection

import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

def mouseEvent(event, x, y, flags, param):
    """
    Defines the functionality for any mouse events
    that occur in the main window.
    """

    if event == cv2.EVENT_MOUSEMOVE:
        print(f"x:{x}, y:{y}")


def trackObject(roi_img, gray_img):
    global roi
    global frame

    # offset to start search from founding position
    offset = 3
    search_increment = 1


    # dimensions to add to the bounding box to check for the image
    margin_of_error = 0

    # get the region of interest coordinates
    roi_col = int(roi[0])
    roi_row = int(roi[1])
    roi_col_width = int(roi[2])
    roi_row_height = int(roi[3])

    # use a hashmap to store distance against the corresponding coordinates 
    distances = {}

    # iterate through neighboring boxes
    for col in range(roi_col - offset, roi_col + offset, search_increment):
        for row in range(roi_row - offset, roi_row + offset, search_increment):
            compare_img = gray_img[row : row + roi_row_height + margin_of_error, col : col + roi_col_width + margin_of_error]
            d1 = cv2.matchShapes(roi_img, compare_img, cv2.CONTOURS_MATCH_I1, 0)
            distances[d1] = (col, row, roi_col_width + margin_of_error, roi_row_height + margin_of_error)


    newImg = frame.copy()

    # get the key with the least distance
    least_distance = min(distances.keys())

    # prevent pingpong effect by not using saame image
    threshold = 0.0001
    if least_distance >= threshold:
        # get the countour that maps to that key
        best_match_countour = distances[least_distance]
        # update roi to that countour
        roi = best_match_countour
        # draw rectangle there
        cv2.rectangle(newImg, (best_match_countour[0], best_match_countour[1]), (best_match_countour[0] + best_match_countour[2], best_match_countour[1] + best_match_countour[3]), (0, 0, 255), 2)
    else:
        # draw rectangle at original roi
        cv2.rectangle(newImg, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 0, 255), 2)
    
    # display the image
    cv2.imshow("frame", newImg)


def main():
    """
    Plays a video from a file.
    """

    # global vars
    global roi
    global frame

    filename = askopenfilename()
    cap = cv2.VideoCapture(filename)

    # store roi and the original image that
    # was at the roi
    roi = ()
    roi_img = np.array([])

    while(True):
        ret, frame = cap.read()

        # store a gray, gaussian blurred version
        # to make comparison better
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if ret:
            # If there is a selected region, track the object
            if roi_img.any():                
                cv2.imshow("selected image", roi_img)
                trackObject(roi_img, gray_frame)
            # otherwise just show the video
            else:
                cv2.imshow("frame", frame)

            key = cv2.waitKey(10)

            # If the q key is pressed, the loop will exit
            if key == ord('q'):
                break
            
            # select a roi via cv2
            if key == ord('x'):
                roi = cv2.selectROI("frame", frame)
                roi_img = gray_frame[int(roi[1]):int(roi[1]+roi[3]),
                      int(roi[0]):int(roi[0]+roi[2])]

        else:
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()

