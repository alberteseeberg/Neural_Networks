
#Import necessary packages

import matplotlib.pyplot as plt
import keras
import numpy as np
import glob
import os
from PIL import Image
import tqdm
import ntpath


#IMG_DIR is the directory where our images are. 
# IM_WIDTH and IM_HEIGHT are the width and height of images after we do some pre-processing.

IMG_DIR = "/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/train/"
IM_WIDTH = 128
IM_HEIGHT = 128


def read_images(directory, resize_to=(128, 128)):
    """
    Reads images and labels from the given directory
    :param directory directory from which to read the files
    :param resize_to a tuple of width, height to resize the images
    : returns a tuple of list of images and labels
    """
    files = glob.glob(directory + "*.jpg")
    images = []
    labels = []
    get_file_name = lambda p: ntpath.split(p)[1]
    
    for f in files:
        im = Image.open(f)
        im = im.resize(resize_to)
        im = np.array(im) / 255.0
        im = im.astype("float32")
        images.append(im)
       
        if get_file_name(f).lower().startswith("dog"):
            label = 1
        else:
            label = 0
        
        labels.append(label)
       
    return np.array(images), np.array(labels)
 
X, y = read_images(directory=IMG_DIR, resize_to=(IM_WIDTH, IM_HEIGHT))
# make sure we have 25000 images if we are reading the full data set.
# Change the number accordingly if you have created a subset
assert len(X) == len(y) == 25000

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# remove X and y since we don't need them anymore
# otherwise it will just use the memory
del X
del y

#Save as npy files
#np.save(
    "/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/images_train.npy"
    , X_train)

#np.save(
    "/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/images_test.npy"
    , X_test)

#np.save(
    "/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/labels_train.npy"
    , y_train)

#np.save(
    "/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/labels_test.npy"
    , y_test)


#labels = np.load("/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/labels.npy")
#images = np.load("/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/images.npy")


#X_train = np.load("/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/images_train.npy")
#X_test = np.load("/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/images_test.npy")
#y_train = np.load("/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/labels_train.npy")
#y_test = np.load("/Users/alberteseeberg/Desktop/Semester_1_kandidat/Advanced_Cognitive_Neuroscience/Neural_Networks/simple-neural-network/kaggle_dogs_vs_cats/labels_test.npy")



#After splitting, we have 17,500 images for training and 7,500 for testing.

X_train.shape, X_test.shape
# ((17500, 128, 128, 3), (7500, 128, 128, 3))


#Checking some images
# Defining a couple of helper functions to visualize images in a grid and also to convert numeric label to string.


def plot_images(images, labels):
    n_cols = min(5, len(images))
    n_rows = len(images) // n_cols
    fig = plt.figure(figsize=(8, 8))
 
    for i in range(n_rows * n_cols):
        sp = fig.add_subplot(n_rows, n_cols, i+1)
        plt.axis("off")
        plt.imshow(images[i])
        sp.set_title(labels[i])
    plt.show()
   
def humanize_labels(labels):
    """
    Converts numeric labels to human friendly string labels
    :param labels numpy array of int
    :returns numpy array of human friendly labels
    """
    return np.where(labels == 1, "dog", "cat")
 
plot_images(X_train[:20], humanize_labels(y_train[:20]))


#Building the model

from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, MaxPool2D
from keras.models import Model
from keras.optimizers import SGD
 
image_input = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
x = Conv2D(filters=32, kernel_size=7)(image_input)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D()(x)
 
x = Conv2D(filters=64, kernel_size=3)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D()(x)
 
x = Conv2D(filters=128, kernel_size=3)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D()(x)
 
x = Flatten()(x)
x = Dense(units=128)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dense(units=1)(x)
x = Activation("sigmoid")(x)
 
opt = SGD(lr=0.001, momentum=0.9)
model = Model(inputs=image_input, outputs=x)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

#Running the model for 10 epochs
model.fit(X_train, y_train, batch_size=64, epochs=10)

#Model evaluation
print(model.metrics_names)
model.evaluate(X_test, y_test, batch_size=128)



predictions = model.predict(X_test)
predictions = np.where(predictions.flatten() > 0.5, 1, 0)
# plot random 20
p = np.random.permutation(len(predictions))
plot_images(X_test[p[:20]], humanize_labels(predictions[p[:20]]))