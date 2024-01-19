import cv2
import numpy as np

# pretrained model files
modelFile = "dnn_model/yolov4-tiny.weights"
configFile = "dnn_model/yolov4-tiny.cfg"
classFile = "dnn_model/classes.txt" 



# OpenCV DNN
net = cv2.dnn.readNet(modelFile, configFile)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)


# check class labels
with open(classFile) as fp:
    labels = fp.read().split("\n")
# print(labels)


# Initialize Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# FULL HD 1920 x 1080


# window name
win_name = "Frame"

# create window
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)


while cv2.waitKey(1) != 27:
    #  The has_frame variable stores the boolean value and the frame variable stores the array.
    has_frame, frame = cap.read()
    if not has_frame:
        break
    
    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=.4)

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        # print(x, y, w, h)
        
        if labels[class_id] == "person":
              cv2.putText(frame, str(labels[class_id]), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
              cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)


    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()