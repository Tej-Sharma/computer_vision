# Filename: mousecallbacks.py
# Author: Tejas Sharma
# Date Created: October 20
# Date Modified: October 20

# This assignment demonstrates the functionality
# of mouse callbacks. Given an image, regions of it can
# be copied and pasted into other parts of the image using
# the mouse position. It also demonstrates the use of character
# input in windows.

import cv2
import numpy as np
from tkinter.filedialog import askopenfilename

def mouseEvent(event,x,y,flags,param):
    """
    This function is called whenever the mouse moves inside
    the window attached to the callback. It stores the current
    mouse position in a global variable. Toggles between copying
    and pasting the region of interest into the image.
    """
    # Declare all of the global variables in this function
    # that were created in main.
    global src
    global modified_img
    global select_size
    global output_win
    global roicopy
    global roipaste
    global roi

    img_rows, img_cols, _ = src.shape

    # Every time the mouse moves, you'll want to draw a rectangle
    # around the center of the mouse's current location. To
    # know whether the mouse is moving, you'll need to check
    # the mouse evenqts. Take a look at OpenCV's HighGUI reference
    # to see the types of mouse events than can occur.
    # https://docs.opencv.org/4.x/d4/dd5/highgui_8hpp.html
    if event == cv2.EVENT_MOUSEMOVE:
        newImg = modified_img.copy()
        cv2.rectangle(newImg, (x - select_size // 2, y - select_size // 2), (x + select_size // 2, y + select_size // 2), (0, 0, 255), 2)
        cv2.imshow(output_win, newImg)

    # The trick here is to make a copy of the image and draw on
    # the copy. If not, you will have tons of squares on the
    # original image.

    # The next thing you'll want to check is if the user has clicked
    # a button. If so, they are either copying or pasting from the
    # clipboard.
    if event == cv2.EVENT_LBUTTONDOWN:
        # If they are pasting a region, you want to copy the
        # region of interest into the image.

        # Otherwise, you'll copy the region into the roi. Every time
        # you copy a segment, you'll also want to display what is
        # being copied into the roi in the Clipboard window.
        
        if not roi.any():
            roi = modified_img[y - select_size // 2 : y + select_size // 2, x - select_size // 2 : x + select_size // 2, :]
            cv2.imshow(clipboard_win, roi)
        else:
            modified_img[y - select_size // 2 : y + select_size // 2, x - select_size // 2 : x + select_size // 2, :] = roi
            roi = np.array([])
            empty_img = np.zeros((select_size,select_size,3), dtype=np.uint8)
            cv2.imshow(clipboard_win, empty_img)
            
        newImg = modified_img.copy()
        cv2.rectangle(newImg, (x - select_size // 2, y - select_size // 2), (x + select_size // 2, y + select_size // 2), (0, 0, 255), 2)
        cv2.imshow(output_win, newImg)


def main():

    # Declare any global variables here
    global src
    global modified_img
    global select_size
    global output_win
    global clipboard_win
    global roicopy
    global roipaste
    global roi

    select_size = 50
    roi = np.array([])

    output_win = "Output Window"
    clipboard_win = "Clipboard Window"

    # Read in the file and store a copy of the original image.
    # Be sure to make a deep copy as we will modify the contents
    # of the original file.
    filename = askopenfilename()

    # Read the image as-is (don't resize, convert to grayscale, etc.)
    # Save the image to a variable called src. This will be the source image.
    src = cv2.imread(filename)
    modified_img = src.copy()

    # Next, used the namedWindow tool to create a window to
    # display your output. You'll also assign the mouse callback
    # to this window. This callback funtion will handle all of 
    # your mouse events. 
    cv2.namedWindow(output_win, cv2.WINDOW_GUI_EXPANDED)
    cv2.setMouseCallback(output_win, mouseEvent)

    # Create a second window for your clipboard. This will keep
    # track of the regions that you are copying and pasting
    # within your image.
    cv2.namedWindow(clipboard_win, cv2.WINDOW_GUI_EXPANDED)

    # Initialize two variables, dx and dy. This will determine
    # the width and height of your region of interest (the area
    # that you are copying and pasting).
    dx, dy = 0, 0

    # In this next section, we will create three bool variables
    # roicopy, roipaste, and roi. We'll use these variables to
    # let us know when we should be copying a region, pasting
    # a region, and the region which we are copying and pasting.
    # Since we cannot pass variables to the mouse callback thread,
    # we will need to create these variables as global variables.
    # This is generally done at the top of the function.

    # Now that everything has been initialized, display the image
    # using imshow.
    cv2.imshow(output_win, src)

    # Using a while loop, we'll continuously listen for input using
    # the mouse callback function/thread and the wawitKey function.
    # Previously, we only used the wait portion of waitKey(); however,
    # we can also use the second part of the function, which returns
    # the value of the key pressed when while the program is waiting.
    while True:

        # Inside the while loop, you are going to listen for 4 diff
        # actions. If the 'q' key is pressed, the loop should end.
        # If the 'c' key is pressed, you'll want to clear the screen
        # and restore the current image to the original image. Pressing
        # 'w' should increase the size of your region of interest and
        # pressing 's' should shrink your region of interest by size
        # of 5.
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord('c'):
            # clear image
            modified_img = src.copy()
        elif key == ord('w'):
            # increase size of region
            select_size += 50
        elif key == ord('s'):
            # decrease size of region
            select_size -= 50
            # if it goes below 0, default to 50
            if select_size <= 0:
                select_size = 50

if __name__ == "__main__":
    main()
