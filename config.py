#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/3/26  
#@Author: Jee
import os

POSITIVE_THRESHOLD = 2000.0



# Directories which contain the positive and negative training image data.
POSITIVE_DIR = './training/positive'
NEGATIVE_DIR = './training/negative'

# Value for positive and negative labels passed to face recognition model.
# Can be any integer values, but must be unique from each other.
# You shouldn't have to change these values.
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 2

# Prefix for positive training image filenames.
POSITIVE_FILE_PREFIX = 'positive_'

# Size (in pixels) to resize images for training and prediction.
# Don't change this unless you also change the size of the training images.
FACE_WIDTH  = 92
FACE_HEIGHT = 112

# base dir
BASE_DIR = os.path.abspath('./')

# Face detection cascade classifier configuration.
# You don't need to modify this unless you know what you're doing.
# See: http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html
HAAR_SCALE_FACTOR  = 1.3
HAAR_MIN_NEIGHBORS = 4
HAAR_MIN_SIZE      = (30, 30)

#training images numbers for every face
FACE_NUM = 60
# Filename to use when saving the most recently captured image for debugging.
# DEBUG_IMAGE = 'capture.pgm'

# Pin Number of Flash light
# FLASH_LIGHT_PIN = 40

#Classifier file
FACE_CLASSIFIER_FILE = BASE_DIR + '/cascades/haarcascade_frontalface_alt.xml'
EYES_CLASSIFIER_FILE = BASE_DIR + '/cascades/haarcascade_eye.xml'
NOSE_CLASSIFIER_FILE = BASE_DIR + '/cascades/haarcascade_mcs_nose.xml'
MOUTH_CLASSIFIER_FILE = BASE_DIR + '/cascades/haarcascade_mcs_mouth.xml'

# Faces dir
# FACES_DIR = BASE_DIR + '/facerec/faces'

#Training dir
TRAINING_DIR = BASE_DIR + '/data'

# File to save and load face recognizer model.
TRAINING_MODEL = BASE_DIR + '/train/trainModel.xml'

#train csv file
TRAINING_CVS_FILE = BASE_DIR + '/train/trainFace.csv'