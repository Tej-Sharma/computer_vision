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

def mouseEvents(event, x, y, flags, params):
    global framenum

    if event == cv2.EVENT_LBUTTONDOWN:
        print('clicked the mouse')

def main():

    global framenum

    win1 = "Frame"
    cv2.namedWindow(win1, cv2.WINDOW_GUI_EXPANDED)
    cv2.setMouseCallback(win1, mouseEvents)

    cap = cv2.VideoCapture(0)

    while True:
        hasframe, frame = cap.read()
        if hasframe:
            cv2.imshow(win1, frame)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
        else:
            break



if __name__ == "__main__":
    main()

