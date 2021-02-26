# Import packages
import numpy as np
import argparse
import time
import cv2
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path to input image")
ap.add_argument("-y", "--yolo", required = True, help = "base path to YOLO directory")
ap.add_argument('-c', "--confidence", type = float, default = 0.5, help = "minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type = float, default = 0.3, help = "threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# Load the COCO class labels our YOLO model was trained on
labels_path = os.path.sep.join([args["yolo"], "coco.names"])
labels = open(labels_path).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
colors = np.random.randint(0, 255, size = (len(labels), 3), dtype = "uint8")

# Derive the paths to the YOLO weights and model configuration
weights_path = os.path.sep.join([args["yolo"], "yolov3.weights"])
config_path = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Load YOLO object detector trained on COCO dataset (80 classes)
print("[[INFO]]  Loading YOLO from disk ...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Load input image and extract spatial dimensions
image = cv2.imread(args["image"])
(height, width) = image.shape[:2]

# Determine only the *output* layer names that we need from YOLO
layer_name = net.getLayerNames()
layer_name = [layer_name[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Construct a blob from the input image and then
# perform a forward pass of the YOLO object detector, giving
# the bounding boxes and associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416,416), swapRB = True, crop = False)
net.setInput(blob)
start = time.time()
layer_outputs = net.forward(layer_name)
end = time.time()

# Show timing information on YOLO
print("[[INFO]]  YOLO took {:.6f} seconds".format(end - start))

# Initialize lists of detected bounding boxes, confidences
# and class IDs, respectively
boxes = []                          # Bounding boxes around object
confidences = []                    # Confidence value YOLO gives to object
classIDs = []                       # Object's class label

# Loop over each of the layer outputs
for output in layer_outputs:

    # Loop over each of the detections
    for detection in output:
        #
        #
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        #
        #
        if confidence > args["confidence"]:
            #
            #
            box = detection[0:4] * np.array([width, height, width, height])
            (centerX, centerY, width, height) = box.astype("int")

            #
            #
            x = int(centerX - (width/2))
            y = int(centerY - (height/2))

            #
            #
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

#
#
indexes = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

#
if len(indexes) > 0 :
    #
    for i in indexes.flatten():
        #
        x = boxes[i][0]
        y = boxes[i][1]
        w = boxes[i][2]
        h = boxes[i][3]

        #
        color = [int(c) for c in colors[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{} : {:.4f}".format(labels[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)


