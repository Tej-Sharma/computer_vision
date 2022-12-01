# Filename: FaceDetection.py
# Author: Jason Grant, Villanova University
# Date Created: 10/21/2021

import numpy as np
import cv2
import sys

def main():

    # The default capture device is the default video source.
    cap = None
    if len(sys.argv) > 1:
        source = sys.argv[1]
        try:
            cap = cv2.VideoCapture(source)
        except:
            sys.exit("Invalid video file.")
    else:
        cap = cv2.VideoCapture(0)

    # Load in the frontal face classifier
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Create a window
    winname = "Faces"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)

    hasframe = True
    while(hasframe):
        # Capture frame-by-frame
        hasframe, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        detected_faces = face_cascade.detectMultiScale(gray)

        # draw a rectangle on the detected faces
        for (column, row, width, height) in detected_faces:
          cv2.rectangle(frame,(column, row),(column + width, row + height),(0, 255, 0),2)
        
        cv2.imshow(winname, frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # When everything's done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
