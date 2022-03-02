import numpy as np
import imutils
import time
import cv2
import os
import math
import os
import sys
from threading import Timer
import shutil
import time
def create_dataset_folders(dataset_path, labels):
    for label in labels:
        dataset_folder = dataset_path + "\\" + label
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
def detect_face(frame, faceNet, threshold=0.5):
    global detections
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    locs = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            locs.append((startX, startY, endX, endY))
    return (locs)
def capture_face_expression(face_expression, label, dataset_path):
    if len(face_expression) != 0:
        dataset_folder = dataset_path + "\\" + label
        number_files = len(os.listdir(dataset_folder))  # dir is your directory path
        image_path = "%s\\%s_%d.jpg" % (dataset_folder, label, number_files)
        cv2.imwrite(image_path, face_expression)
dataset_path = os.getcwd() + "\\dataset"
face_model_path = os.getcwd() + "\\face_detector"
labels = ["neutral", "happy", "sad", "angry"]
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([face_model_path, "deploy.prototxt"])
weightsPath = os.path.sep.join([face_model_path, "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print("[INFO] Creating dataset folders...")
create_dataset_folders(dataset_path, labels)
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    locs = detect_face(frame, faceNet, threshold=0.5)
    face_expression = None
    for box in locs:
        (startX, startY, endX, endY) = box
        face_expression = gray[startY:endY, startX:endX].copy()
        cv2.rectangle(gray, (startX, startY), (endX, endY), (255, 255, 255), 2)
    cv2.imshow('Data Set Creation', gray)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('n'):
        capture_face_expression(face_expression, "neutral", dataset_path)
        print("neutral")
    elif key == ord('h'):
        capture_face_expression(face_expression, "happy", dataset_path)
        print("happy")
    elif key == ord('s'):
        capture_face_expression(face_expression, "sad", dataset_path)
        print("sad")
    elif key == ord('a'):
        capture_face_expression(face_expression, "angry", dataset_path)
        print("angry")
cap.release()
cv2.destroyAllWindows()
