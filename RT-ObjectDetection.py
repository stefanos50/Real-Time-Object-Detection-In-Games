import pygame
import os
import random
import numpy as np
import argparse
import time
import cv2 as cv
import os
import pyautogui
import pygetwindow
from pynput.keyboard import Key, Controller
import time
import pygetwindow as gw
from PIL import ImageGrab, Image

keyboard = Controller()


#keyboard.press(Key.space)
distance_threshold = 490
# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
path = "E:\AdvancedLocomotionSystemV\Saved\Screenshots\WindowsEditor\screenshot2.jpg"
def get_frame():
    try:
        window = pygetwindow.getWindowsWithTitle("AdvancedLocomotionV4 AdvancedLocomotionV4")[0]
    except:
        print('Did not found the game window...')
        return -1
    left,top = window.topleft
    right,bottom = window.bottomright
    pyautogui.screenshot(path)
    try:
        # Capture the screen region occupied by the game window
        #screenshot = ImageGrab.grab(bbox=(left+10,top+30,right-10,bottom-10))

        # Save the screenshot as an image
        #screenshot.save(path)
        im = Image.open(path)
        im = im.crop((left+10,top+30,right-10,bottom-10))
        im.save(path, compress_type=3)
        #image.save("tmp1.png", compress_type=3)
    except:
        print("Cant save frame...Skipping...")
        return

def calculate_euclidean_distance(v1,v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    temp_v = v1 - v2

    euclid_distance = np.sqrt(np.dot(temp_v.T,temp_v))

    return euclid_distance

def calculate_distance(entities,bb):
    try:
        person_id = entities.index('person')
        person_bb = bb[person_id]
    except:
        print("Not found a person...")
        return -1,""

    found_object = 0
    found_object_key = 'w'
    try:
         obj_id = entities.index('car')
         obj_bb = bb[obj_id]
         found_object_key = 'w'
    except:
        print("Not found a car...")
        found_object = -1,""
        try:
            obj_id = entities.index('bird')
            obj_bb = bb[obj_id]
            found_object_key = 's'
        except:
            print("Not found a bird...")
            found_object = -1
        if found_object == -1:
            return -1, ""
    #print(calculate_euclidean_distance([person_bb[2],person_bb[3]],[car_bb[0],car_bb[1]]))
    return calculate_euclidean_distance([person_bb[2],person_bb[3]],[obj_bb[0],obj_bb[1]]),found_object_key

def press_action_key(key):
    time.sleep(0.001)
    keyboard.press(key)
    time.sleep(0.001)
    keyboard.release(key)


def run_ssd():
    image = cv.imread(path)
    try:
        (h, w) = image.shape[:2]
    except:
        print("Cant Find Frame... Skipping...")
        return
    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 0.007843,
                                 (300, 300), 127.5)
    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    detected_entities = []
    detected_bb = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.7:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            detected_entities.append(CLASSES[idx])
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            detected_bb.append([startX, startY, endX, endY])
            print("[INFO] {}".format(label))
            cv.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv.putText(image, label, (startX, y),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    frame_result,key_result = calculate_distance(detected_entities,detected_bb)
    if frame_result != -1:
        print(frame_result)
        if frame_result < distance_threshold:
            press_action_key(key_result)
    cv.imshow("Output", image)
    cv.waitKey(1)




while True:
    r = get_frame()
    if r != -1:
        run_ssd()
