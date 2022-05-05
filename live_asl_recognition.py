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

IMG_WIDTH=200
IMG_HEIGHT=200


# model = load_model('trained_model')
# model = load_model('edge_detection_model')
# model = load_model('binary_threshold_model')
# model = load_model('grayscale_model')
model = load_model('vowel_thresholding_model')

widthCam, heightCam = 640, 480

# create VideoCapture object with camera as input (0)
cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)
prevTime = 0 #initializing previous time
curTime = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

       
while(True):
    ret,img=cap.read() # grabbing each frame of webcam footage
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img,(IMG_WIDTH, IMG_HEIGHT)) # resizing frame size, must match video output frame size

    '''MediaPipe Code'''
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    '''Thresholding/Edge Detection Processing'''

    cv2.rectangle(img, (50,50),(400,400),(0,255,0),2) # adding a rectangle to the image
    frame_img = cv2.resize(img[50:400,50:400],(IMG_WIDTH,IMG_HEIGHT))
    
    gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (3,3), 0)

    # # sobel_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    ret, binary_img = cv2.threshold(img_blur, 80, 255, cv2.THRESH_BINARY_INV)

    # frame_img = np.expand_dims(frame_img, axis=0)
    # sobel_img = np.expand_dims(sobel_img, axis=0)
    binary_img = np.expand_dims(binary_img, axis=0)
    # print(frame.shape)

    # prediction = model.predict(frame_img)
    # prediction = model.predict(sobel_img)
    prediction = model.predict(binary_img)

    predicted_label = np.argmax(prediction, axis=1)
    predicted_letter = list(letters_with_labels.keys())[list(letters_with_labels.values()).index(predicted_label)]
    # print(predicted_letter) # live prediction of frame
    # print(percent) # confidence
    # cv2.putText(img,'{} {:.2f}% res50'.format(predicted_label),(25),cv2.FONT_HERSHEY_SIMPLEX,0.9,(80,255,255),1)
    
    font=cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, predicted_letter, (30,30), font, fontScale=1, color=(255,0,0), thickness=2)
    # out.write(img) # output frame
    cv2.imshow('Letter', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


# SAM'S CODE


# widthCam, heightCam = 640, 480

# # create VideoCapture object with camera as input (0)
# cap = cv2.VideoCapture(0)
# cap.set(3, widthCam)
# cap.set(4, heightCam)
# prevTime = 0 #initializing previous time
# curTime = 0

# mpHands = mp.solutions.hands
# hands = mpHands.Hands(static_image_mode=False,
#                       max_num_hands=2,
#                       min_detection_confidence=0.5,
#                       min_tracking_confidence=0.5)
# mpDraw = mp.solutions.drawing_utils


# while True:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#     #print(results.multi_hand_landmarks)
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             for id, lm in enumerate(handLms.landmark):
#                 #print(id,lm)
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x *w), int(lm.y*h)
#                 #if id ==0:
#                 cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
#             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

#     imgRGB = cv2.resize(imgRGB, (IMG_WIDTH, IMG_HEIGHT))
#     imgRGB = np.expand_dims(imgRGB, axis=0)
#     prediction = model.predict(imgRGB)
#     predicted_label = np.argmax(prediction, axis=1)
#     predicted_letter = list(letters_with_labels.keys())[list(letters_with_labels.values()).index(predicted_label)]
#     print(predicted_letter) # live prediction of frame
#     font = cv2.FONT_HERSHEY_SIMPLEX
    
#     cv2.putText(img, predicted_letter, (50,50), font, fontScale=1, color=(255,0,0), thickness=2)
    
#     curTime = time.time() # current time
#     fps = 1/(curTime - prevTime)
#     prevTime = curTime
    
#     cv2.imshow("Img", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

