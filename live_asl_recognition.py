import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time
import mediapipe as mp

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
letters_with_labels = {letter : i for i, letter in enumerate(letters)}
# print(letters_with_labels)

# Defining image sizes
IMG_WIDTH=200
IMG_HEIGHT=200

# Choose which model you would like to load
# model = load_model('trained_model')
# model = load_model('edge_detection_model')
# model = load_model('binary_threshold_model')
# model = load_model('grayscale_model')
model = load_model('vowel_thresholding_model')

# Setting width and height of webcam
widthCam, heightCam = 640, 480

# create VideoCapture object with camera as input (0)
cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)

#initializing previous time
prevTime = 0
curTime = 0

# Creating media pipe object
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

       
while(True):
    # grabbing each frame of webcam footage
    ret,img=cap.read() 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    '''MediaPipe Code'''
    # Drawing lines and circles to track hand
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    '''Threshold Image Processing'''
    # Adding rectangle to image 
    cv2.rectangle(img, (50,50),(400,400),(0,255,0),2)
    # Capturing image within rectangle and resizing it accordingly 
    frame_img = cv2.resize(img[50:400,50:400],(IMG_WIDTH,IMG_HEIGHT))
    
    # Image conversion to grayscale and smoothing
    gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (3,3), 0)

    # sobel_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    # Applying binary threshold to image
    ret, binary_img = cv2.threshold(img_blur, 80, 255, cv2.THRESH_BINARY_INV)

    # Implementing shape correction
    # frame_img = np.expand_dims(frame_img, axis=0)
    # sobel_img = np.expand_dims(sobel_img, axis=0)
    binary_img = np.expand_dims(binary_img, axis=0)


    # Predicting image from model
    # prediction = model.predict(frame_img)
    # prediction = model.predict(sobel_img)
    prediction = model.predict(binary_img)

    # Acquire correct letter from predicted value from dictionary
    predicted_label = np.argmax(prediction, axis=1)
    predicted_letter = list(letters_with_labels.keys())[list(letters_with_labels.values()).index(predicted_label)]

    # Outputting predicted letter to screen    
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, predicted_letter, (30,30), font, fontScale=1, color=(255,0,0), thickness=2)

    cv2.imshow('Letter', img)

    # Press 'q' to quit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()