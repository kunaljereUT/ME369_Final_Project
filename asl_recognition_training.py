import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.utils import shuffle

import numpy as np
import os
import cv2

# Defining image sizes
IMG_WIDTH=200
IMG_HEIGHT=200

# Initializing Letters and Vowels
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
vowels = ['A', 'E', 'I', 'O', 'U', 'Y']
letters_with_labels = {letter : i for i, letter in enumerate(letters)}
# print(letters_with_labels)

# Path to directory that holds image folders
dir = r'C:\Users\kunal\Documents\ME369P\Final_Project'

def load_training_data():
    # Specific folder to load data from
    category = 'asl_alphabet_train'

    # Connects directory and category into single path
    path = os.path.join(dir, category)

    print('Loading {}'.format(category))

    train_images = []
    train_labels = []

    # Loops through each folder in training folder
    for folder in os.listdir(path):
        if (folder in vowels):

            # Assigning numeric label based on letter of folder
            label = letters_with_labels[folder]
            
            # Looping through each image file in alphabet folder
            for idx, file in enumerate(os.listdir(os.path.join(path, folder))):
                # Number of image we want to train on, change value based on memory allocation
                if (idx == 1001):#801
                    break
                
                # Getting path of each image
                img_path = os.path.join(os.path.join(path, folder), file)

                # Opening and resizing images
                img = cv2.imread(img_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                # Converting image to grayscale and smoothing them out
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(gray, (3,3), 0)

                # Applying binary threshold to each image
                ret, binary = cv2.threshold(img_blur, 80, 255, cv2.THRESH_BINARY_INV)
                # sobel_img = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

                train_images.append(binary)
                train_labels.append(label)
            
        else:
            continue

    # Converting list of images and labels to numpy array
    train_images = np.array(train_images, dtype='float32')
    train_labels = np.array(train_labels, dtype='int32')
    
    # next 2 lines only needed if training grayscale model
    # numberImages = np.size(train_images, 0)
    # train_images = train_images.reshape(numberImages, IMG_WIDTH, IMG_HEIGHT, 1)

    
    return (train_images, train_labels)


train_images, train_labels = load_training_data()
# shuffling image and labels for randomization
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)

# Creation of simple CNN model
model = tf.keras.Sequential([
    Conv2D(32, (5,5), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(2,2),
    Conv2D(32, (5,5), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
fitted_model = model.fit(train_images, train_labels, batch_size=100, epochs=3, validation_split=0.2)

# Saving model in directory
model.save('vowel_thresholding_model')
