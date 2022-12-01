# Author: Jason Grant, Villanova University
# Read Ground Truth Files
# CSC 5930/9010 - Computer Vision
# MOT Challenge
# Date Created: December 2, 2021

import cv2
import numpy as np
import sys
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-g', '--groundtruth', required=True,
                help = 'path to ground truth file')
ap.add_argument('-i', '--image', required=True,
              help = 'path to image file')        
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
args = ap.parse_args()

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    if label != "person":
      return

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    print(classes)

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)


def yolo_tracker(image, winname):
  scale = 0.00392
  blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
  net.setInput(blob)
  outs = net.forward(get_output_layers(net))
  class_ids = []
  confidences = []
  boxes = []
  conf_threshold = 0.5
  nms_threshold = 0.4
  Width = image.shape[1]
  Height = image.shape[0]

  for out in outs:
      for detection in out:
          scores = detection[5:]
          class_id = np.argmax(scores)
          confidence = scores[class_id]
          if confidence > 0.5:
              center_x = int(detection[0] * Width)
              center_y = int(detection[1] * Height)
              w = int(detection[2] * Width)
              h = int(detection[3] * Height)
              x = center_x - w / 2
              y = center_y - h / 2
              class_ids.append(class_id)
              confidences.append(float(confidence))
              boxes.append([x, y, w, h])


  indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

  for i in indices:
      try:
          box = boxes[i]
      except:
          i = i[0]
          box = boxes[i]
      
      x = box[0]
      y = box[1]
      w = box[2]
      h = box[3]
      draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
  
  cv2.imshow(winname, image)

def getboxes(filename):
    """
    Reads in the ground truth file and returns a dictionary
    key of the dictionary is the frame and the value is a
    list of all bounding boxes in the frame
    Note: other data fields ignored
    """

    boxes = dict()

    with open(filename) as f:
        lines = f.readlines()

        for line in lines:
            data = line.split(',')
            frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z = data

            frame = int(frame)
            id = int(id)
            bb_left = int(bb_left)
            bb_top = int(bb_top)
            bb_width = int(bb_width)
            bb_height = int(bb_height)

            if frame not in boxes:
                boxes[frame] = [(bb_left, bb_top, bb_width, bb_height)]
            else:
                boxes[frame].append( (bb_left, bb_top, bb_width, bb_height) )

    return boxes


def my_tracker(grey, frame, winname):
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
  detected_objects = face_cascade.detectMultiScale(grey)
  for (column, row, width, height) in detected_objects:
          cv2.rectangle(frame,(column, row),(column + width, row + height),(0, 255, 0),2)
  cv2.imshow(winname, frame)

def main():

    if len(sys.argv) < 3:
        print(f'{sys.argv[0]} [groundtruth] [img dir]')
        sys.exit()

    groundtruth = args.groundtruth
    dir = args.image

    # Read in the file and get all of the bounding box data
    bb = getboxes(groundtruth)

    cv2.namedWindow("Ground Truth", cv2.WINDOW_NORMAL)

    framenum = 1
    winname = "My Detector"

    with os.scandir(dir) as it:

        # sort the files in alphabetical order so the video
        # appears in sequential order
        it = list(it)
        it.sort(key=lambda x: x.name)

        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():

                # Read in the image
                frame = cv2.imread(dir + "/" + entry.name)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                yolo_tracker(frame, winname)

                # # for each frame, go through the dictionary and find
                # # the corresponding bounding boxes for each frame
                # # Then, draw them on the frame
                # boxes = bb[framenum]
                # for box in boxes:
                #     cv2.rectangle(src, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (0,0,255), 2)

                # cv2.imshow("Ground Truth", src)
                
                cv2.waitKey(30)

                framenum += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
