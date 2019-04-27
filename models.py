'''
File    : models.py
Author  : Jay Santokhi (jks1g15)
Brief   : Contains functions for training
'''
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Dense, Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import np_utils

from sklearn.metrics import classification_report
from sklearn.cluster import MiniBatchKMeans
from numpy import newaxis
from sklearn import svm

import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2


def get_patches(image):
    ''' Returns an array of 2x2 patches in an image

    Arguments:
        image -- Desired image to take patches from

    Returns:
        mc -- Mean Centred patches
    '''
    patches = []
    for y in range(0, image.shape[0] - 2, 2):
        for x in range(0, image.shape[1] - 1, 1):
            patches.append(image[y:y + 2, x:x + 2].flatten())
    mc = mean_centre(patches)
    return mc


def mean_centre(patches):
    ''' Mean centre the patches before quantisation

    Arguments:
        patches -- Desired patches to be mean centred

    Returns:
        mc_patches -- Mean Centred Patches
    '''
    mc_patches = []
    for patch in patches:
        mc_patches.append(patch - np.mean(patch))
    return mc_patches


def BoVW_classifier(trainX, trainY, testX, testY):
    ''' Bag of Visual Words Classifier

    Arguments:
        trainX -- Training Data
        trainY -- Training Labels
        testX --  Test Data
        testY --  Test Labels
    '''
    # Get rid of last dimension so its (no of data, height, width)
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2])

    # Get patches from image
    kmeans = MiniBatchKMeans(n_clusters=500, batch_size=100, random_state=42)
    print('Generating patches... ')
    for image in trainX:
        patches = get_patches(image)
        kmeans.partial_fit(patches)
    print('Patches Generated')

    # Generate training histograms
    train_hist = []
    print('Generating training histograms...')
    for image in trainX:
        patches = get_patches(image)
        predictions = kmeans.predict(patches)
        (histogram, bin) = np.histogram(predictions, bins=500)
        train_hist.append(histogram)
    print('Histograms Generated')

    # Train the classifier
    print('Training classifier... ')
    classifier = svm.LinearSVC(multi_class='ovr', max_iter=1000, dual=False)
    classifier.fit(train_hist, trainY)
    print('Classifier Trained')

    # Load test data
    test_hist = []
    print('Generating testing histograms... ')
    for image in testX:
        patches = get_patches(image)
        predictions = kmeans.predict(patches)
        (histogram, bin) = np.histogram(predictions, bins=500)
        test_hist.append(histogram)
    print('Testing Histograms Generated')

    # Predict test data classification
    print('Making Predictions... ')
    pred = classifier.predict(test_hist)
    print('Finished')

    print(classification_report(testY, pred))
    return


def spectrogram_model(num_classes=8):
    ''' Defines model for classifying spectrogram of respiratory sounds

    Arguments:
        num_classes -- Number of classes in the problem

    Returns:
        m -- The model defined in this function
    '''
    m = Sequential()
    #m.add(Conv2D(64,(15,16),padding="same",input_shape=[101,99,1],strides=6))
    # m.add(Conv2D(64,(11,15),padding="valid",input_shape=[101,149,1],strides=6))
    m.add(Conv2D(64,(5,5),padding="valid",input_shape=[186,13,1],strides=3))
    m.add(Dropout(0.2))
    #m.add(BatchNormalization())
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    #m.add(MaxPooling2D(pool_size=(2,2), strides=None))
    m.add(MaxPooling2D(pool_size=(2,2), strides=None))
    m.add(Flatten())
    m.add(Dropout(0.5))
    m.add(Dense(64))
    m.add(Activation('relu'))
    m.add(Dense(num_classes))
    m.add(Activation('softmax'))
    return m


def MFCC_model(num_classes=8):
    ''' Defines model for classifying MFCC Heat Map of respiratory sounds

    Arguments:
        num_classes -- Number of classes in the problem

    Returns:
        m -- The model defined in this function
    '''
    m = Sequential()
    #m.add(Conv2D(64,(1,20),padding="same",input_shape=[499,12,1]))
    m.add(Conv2D(64,(6,6),padding = "valid", input_shape=[499,6,1]))
    m.add(Dropout(0.5))
    m.add(Activation('relu'))
    m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(20,1)))

    #m.add(Conv2D(64,(5,1), padding = "same"))
    #m.add(Dropout(0.5))
    #m.add(Activation('relu'))
    #m.add(BatchNormalization())
    #m.add(MaxPooling2D(pool_size=(5,1)))

    m.add(Flatten())
    m.add(Dropout(0.5))

    #m.add(Dense(512))
    #m.add(Activation('relu'))
    #m.add(Dropout(0.5))

    m.add(Dense(64))
    m.add(Activation('relu'))
    m.add(Dropout(0.5))
    m.add(Dense(num_classes))
    m.add(Activation('softmax'))
    return m


def NN_classifier(trainX, trainY, testX, testY, type):
    ''' Neural Network Classifier

    Arguments:
        trainX -- Training Data
        trainY -- Training Labels
        testX --  Test Data
        testY --  Test Labels
    '''
    trainY = np_utils.to_categorical(trainY,num_classes=8)
    testY = np_utils.to_categorical(testY,num_classes=8)

    if type == 'MFCC':
        model = MFCC_model(num_classes=8)
    elif pp == 'Spectrogram':
        model = spectrogram_model(num_classes=8)

    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

    print(model.summary())

    # If accuracy does not increase over 10 epochs, we reduce the learning rate by half.
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_lr=0.0001, verbose=1)
    batch_size = 64

    # Checkpoint for saving model when validation accuracy increases
    # filepath="Model-{epoch:02d}-{val_acc:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
    #                              save_weights_only=False, mode='max', period=1)

    print('Training network...')
    H = model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=30,
                  verbose=1, huffle=True, validation_data=(testX, testY),
                  callbacks=[reduce_lr])

    print('Saving model...')
    model.save('Model.hdf5')

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 30), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 30), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 30), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, 30), H.history["val_acc"], label="val_acc")
    if type == 'MFCC':
        plt.title("MFCC Training Loss and Accuracy")
    elif pp == 'Spectrogram':
        plt.title("Spectrogram Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
