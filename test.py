# from asl_recognition import test_images, test_labels
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
letters_with_labels = {letter : i for i, letter in enumerate(letters)}
print(letters_with_labels)

IMG_WIDTH=200
IMG_HEIGHT=200

def load_test_data():
    dir = r'C:\Users\kunal\Documents\ME369P\Final_Project\asl_alphabet_test'
    print('Loading asl_alphabet_test')

    test_images = []
    test_labels = []
    for file in os.listdir(dir):
        label = letters_with_labels[str(file[0])]

        img_path = os.path.join(dir, file)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype('float32')
        img /= 255
        
        test_images.append(img)
        test_labels.append(label)

    test_images = np.array(test_images, dtype='float32')
    test_labels = np.array(test_labels, dtype='int32')

    return test_images, test_labels

test_images, test_labels = load_test_data()
test_images, test_labels = shuffle(test_images, test_labels, random_state=25)


model = load_model('trained_model')

eval = model.evaluate(test_images, test_labels)
print('Evaluation Accuracy: ' + str(eval[1]))

class_predictions = model.predict(test_images)
# print('Predicitons:')
# print(class_predictions)

predicted_label = np.argmax(class_predictions, axis=1)
# print('Classfications of Class Predictions:')
# print(classification_report(test_labels, class_predictions))

# print('Classfications of Predicted Labels:')
# print(classification_report(test_labels, predicted_label))
print('Actual Labels:')
print(test_labels)
print('Predicted Labels:')
print(predicted_label)

actual_letters = []
predicted_letters = []
for i in range(len(test_labels)):
    actual_letter = list(letters_with_labels.keys())[list(letters_with_labels.values()).index(test_labels[i])]
    predicted_letter = list(letters_with_labels.keys())[list(letters_with_labels.values()).index(predicted_label[i])]

    actual_letters.append(actual_letter)
    predicted_letters.append(predicted_letter)

#Visualize predictions
L = 7
W = 4

fig, axes = plt.subplots(nrows=L, ncols=W, figsize = (15,15))
axes = axes.ravel()

for i in np.arange(0, 26):  
    axes[i].imshow(test_images[i])#.reshape(3,200,200))
    # axes[i].set_title(f"Prediction Class = {predicted_label[i]:0.1f}\n True Class = {test_labels[i]:0.1f}")
    axes[i].set_title(f"Prediction Class = {predicted_letters[i]}\n True Class = {actual_letters[i]}")
    axes[i].axis('off')
plt.subplots_adjust(wspace=0.5)
plt.show()