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
vowels = ['A', 'E', 'I', 'O', 'U', 'Y']
letters_with_labels = {letter : i for i, letter in enumerate(letters)}
# print(letters_with_labels)

IMG_WIDTH=200
IMG_HEIGHT=200

def load_test_data():
    # dir = r'C:\Users\kunal\Documents\ME369P\Final_Project\asl_alphabet_test'
    # dir = r'C:\Users\kunal\Documents\ME369P\Final_Project\kj_test'
    # dir = r'C:\Users\kunal\Documents\ME369P\Final_Project\sm_test'
    # dir = r'C:\Users\kunal\Documents\ME369P\Final_Project\sp_test'
    dir = r'C:\Users\kunal\Documents\ME369P\Final_Project\custom_test'
    print('Loading Test Images')

    original_images = []
    test_images = []
    test_labels = []
    for file in os.listdir(dir):
        # if (str(file[0]) in vowels):
        label = letters_with_labels[str(file[0])]

        img_path = os.path.join(dir, file)

        img = cv2.imread(img_path)

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_img = orig_img.astype('float32')
        orig_img /= 255
        original_images.append(orig_img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(gray, (3,3), 0)
        
        ret, binary = cv2.threshold(img_blur, 80, 255, cv2.THRESH_BINARY_INV)

        test_images.append(binary)
        test_labels.append(label)
            
        # else:
        #     continue

    test_images = np.array(test_images, dtype='float32')
    original_images = np.array(original_images, dtype='float32')
    test_labels = np.array(test_labels, dtype='int32')

    return test_images, original_images, test_labels

test_images, original_images, test_labels = load_test_data()
test_images, original_images, test_labels = shuffle(test_images, original_images, test_labels, random_state=25)

#Get 28 random images with their labels from custom test folder
num_custom_images = 26*2*3
random_idx = np.random.choice(np.arange(num_custom_images), 28, replace=False)
random_orig_images = original_images[random_idx]
random_test_images = test_images[random_idx]
random_test_labels = test_labels[random_idx]

# next 2 lines only needed to create grayscale_model
# numberImages = np.size(test_images, 0)
# train_images = test_images.reshape(numberImages, IMG_WIDTH, IMG_HEIGHT, 1)

'''Models to Load'''
# model = load_model('trained_model')
# model = load_model('edge_detection_model')
model = load_model('binary_threshold_model')
# model = load_model('grayscale_model')
# model = load_model('vowel_thresholding_model')


# eval = model.evaluate(test_images, test_labels)
eval = model.evaluate(random_test_images, random_test_labels)
print('Evaluation Accuracy: ' + str(eval[1]))


# class_predictions = model.predict(test_images)
class_predictions = model.predict(random_test_images)


predicted_label = np.argmax(class_predictions, axis=1)



actual_letters = []
predicted_letters = []
# for i in range(len(test_labels)):
#     actual_letter = list(letters_with_labels.keys())[list(letters_with_labels.values()).index(test_labels[i])]
#     predicted_letter = list(letters_with_labels.keys())[list(letters_with_labels.values()).index(predicted_label[i])]

#     actual_letters.append(actual_letter)
#     predicted_letters.append(predicted_letter)

for i in range(len(random_test_labels)):
    actual_letter = list(letters_with_labels.keys())[list(letters_with_labels.values()).index(random_test_labels[i])]
    predicted_letter = list(letters_with_labels.keys())[list(letters_with_labels.values()).index(predicted_label[i])]

    actual_letters.append(actual_letter)
    predicted_letters.append(predicted_letter)


print('Actual Labels:')
print(actual_letters)
print('Predicted Labels:')
print(predicted_letters)


#Visualize predictions
L = 7
W = 4

fig, axes = plt.subplots(nrows=L, ncols=W, figsize = (15,15))
axes = axes.ravel()

for i in np.arange(0, 28):  #26
    # axes[i].imshow(original_images[i])#.reshape(3,200,200))
    axes[i].imshow(random_orig_images[i])#.reshape(3,200,200))
    # axes[i].set_title(f"Prediction Class = {predicted_label[i]:0.1f}\n True Class = {test_labels[i]:0.1f}")
    axes[i].set_title(f"Prediction Class = {predicted_letters[i]}\n True Class = {actual_letters[i]}", fontsize=12)
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.5, hspace=1)
plt.show()