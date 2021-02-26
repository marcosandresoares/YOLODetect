# Import packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "path to input video")
ap.add_argument("-o", "--output", required = True, help = "path to output video")
ap.add_argument("-y", "--yolo", required = True, help = "base path to YOLO directory")
ap.add_argument("-c", "--confidence", type = float, default=0.5, help = "minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type = float, default=0.3, help = "threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Load the COCO class labels the model was trained on
labels_path = os.path.sep.join([args["yolo"], "coco.names"])
labels = open(labels_path).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
colors = np.random.randint(0, 255, size = (len(labels), 3), dtype = "uint8")

# Derive paths to the YOLO weights and model configuration
weights_path = os.path.sep.join([args["yolo"], "yolov3.weights"])
config_path = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Load YOLO object detector trained on COCO dataset
# and determine only the *output* layer names that we need from YOLO
print("[[INFO]]  Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_name = net.getLayerNames()
layer_name = [layer_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize the video stream, pointer to output video file, and
# frame dimensions
video_stream = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# Try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
    total = int(video_stream.get(prop))
    print("[[INFO]]  {} total frames in video".format(total))

# an error occurred while trying to determine the total 
# number of frames in the video file
except:
    print("[[INFO]]  Could not determine the number of frames in video")
    print("[[INFO]]  No approximate completion time can be provided")
    total = -1

# Loop over frames from the video file stream
while True:
    
    # Read the next frame from the file
    (grabbed, frame) = video_stream.read()

    # If frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break
    
    # If the frame dimensions are empty, grab the dimensions
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Construct a blob from the input frame and then
    # perform forward pass of the YOLO object detector, 
    # giving the bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(layer_name)
    end = time.time()

    # Initialize lists of detected bounding boxes, confidences, and classIDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layer_outputs:

        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > args["confidence"]:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    if len(indexes) > 0 :

        for i in indexes.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            #
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{} : {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            #
    if writer is None:
                
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        #
        if total > 0 :
            elapsed_time = (end - start)
            print("[[INFO]]  Single Frame took {:.4f} seconds".format(elapsed_time))
            print("[[INFO]]  Estimated Total Time to finish: {:.4f}".format(elapsed_time * total))
                
    #
    writer.write(frame)

# Release the file pointers
print("[[INFO]]  Cleaning Up...")
writer.release()
video_stream.release()