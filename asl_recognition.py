import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

import numpy as np
import os
import cv2

IMG_WIDTH=200
IMG_HEIGHT=200


letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
letters_with_labels = {letter : i for i, letter in enumerate(letters)}
print(letters_with_labels)

dir = r'C:\Users\kunal\Documents\ME369P\Final_Project'

def load_training_data():
    category = 'asl_alphabet_train'

    path = os.path.join(dir, category)

    print('Loading {}'.format(category))

    train_images = []
    train_labels = []

    for folder in os.listdir(path):
        label = letters_with_labels[folder]
        for idx, file in enumerate(os.listdir(os.path.join(path, folder))):
            #Getting path of each image
            if (idx == 801):
                break

            img_path = os.path.join(os.path.join(path, folder), file)

            #Opening and resizing images
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype('float32')
            img /= 255 #normalizing values of pixels

            train_images.append(img)
            train_labels.append(label)

    train_images = np.array(train_images, dtype='float32')
    train_labels = np.array(train_labels, dtype='int32')

    return (train_images, train_labels)


train_images, train_labels = load_training_data()

train_images, train_labels = shuffle(train_images, train_labels, random_state=25)


model = tf.keras.Sequential([
    Conv2D(32, (5,5), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),
    Conv2D(32, (5,5), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
fitted_model = model.fit(train_images, train_labels, batch_size=100, epochs=5, validation_split=0.2)

model.save('trained_model')
