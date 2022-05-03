import numpy as np
import math as m
import cv2
import mediapipe as mp
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# class handDectetor():
#     def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
#         self.mode = mode
#         self.maxHands = maxHands
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon
        
#         self.mpHands = mp.solutions.hands
#         self hands = 

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


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.muti_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    
    curTime = time.time() # current time
    fps = 1/(curTime - prevTime)
    prevTime = curTime
    
    cv2.imshow("Img", img)
    cv2.waitKey(1)
    
    
