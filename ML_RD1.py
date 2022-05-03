import numpy as np
import math as m
import cv2
import mediapipe as mp
import tensorflow as tf
import time
import matplotlib.pyplot as plt


widthCam, heightCam = 640, 480

# create VideoCapture object with camera as input (0)
cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)
prevTime = 0 #initializing previous time

while True:
    success, img = cap.read()
    
    curTime = time.time() # current time
    fps = 1/(curTime - prevTime)
    prevTime = curTime
    
    cv2.imshow("Img", img)
    cv2.waitKey(1)
    