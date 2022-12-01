# Author: 
# Villanova University
# Date Created: 11/17/2022
# 
# Local Binary Patterns
import numpy as np
import cv2

def takephoto():
    """ Capture one image of yourself """

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while(True):
        
        ret, frame = cap.read()
        img = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mcol, mrow, mwid, mheight, marea = (0,0,0,0,0)

        # Detect the faces in the image
        detected_faces = face_cascade.detectMultiScale(gray)

        # Draws the faces on the image and keeps track of the largest face/box
        for (column, row, width, height) in detected_faces:
            cv2.rectangle(img,(column, row),(column + width, row + height),(0, 255, 0),2)
            area = width * height
            if area > marea:
                mcol, mrow, mwid, mheight, marea = (column,row,width,height,area)

        # Makes a copy of the largest, detected face
        face = frame[mrow:mrow+mheight, mcol:mcol+mwid].copy()
        cv2.imshow('Frame',img)
    
        key = cv2.waitKey(1) & 0xFF

        # Press the 's' key to save the image/face and return it to the main program.
        if key == ord('s'):

            cap.release()
            cv2.destroyAllWindows()
            return face

        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return np.zeros((10,10),np.uint8)


def lbp_descriptor(src):
    """
    Computes the Local Binary Pattern descriptor for the source array.
    Output should be a single value.
    """

    pass

def lbp_feature(src):
    """
    Computes the Local Binary Pattern feature for a given image. The
    image should be divided into 4x4 regions and the histogram should
    be computed for each region. The histogram should contain 8 bins,
    and the final vector should be a 16 * 8 array, which is the feature
    LBP feature vector.
    """

    pass

def compare_features(v1, v2):
    """
    Calculates the distance score between two LBP feature vectors
    """



def main():

    # Captures your photo
    face = takephoto()
    cv2.imshow("Template",face)
    cv2.waitKey(0)

    # Compute the LBP feature for your template
    template = lbp_feature(face)
    
    # After you have the template, you want to start the video
    # again and compare your face to the ones on the screen.
    # Draw a green box if the face is yours, draw a red box
    # if the face is someone elses.
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    while(True):
        
        ret, frame = cap.read()
        img = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        # Detect faces. You'll want to call the LBP feature generator on all of the faces
        # found here. If the comparison is below a threshold, consider it a match. Color
        # this box green. If it is not a match, color it red.
        detected_faces = face_cascade.detectMultiScale(gray)
        for (column, row, width, height) in detected_faces:
            ismatch = False
            # Create the LBP vector and compare with the template. If the score is 
            # below a threshold, then it is a match. Otherwise, it is a non-match.
            # Feel free to modify the boolean variable that I have used as a template
            # to get your started.
            if ismatch:
                cv2.rectangle(img,(column, row),(column + width, row + height),(0, 255, 0),2)
            else:
                cv2.rectangle(img,(column, row),(column + width, row + height),(255, 0, 0),2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
