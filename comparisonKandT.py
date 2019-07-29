import os
import argparse
import time

import cv2
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.platform import gfile
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--photo", required=True, help="path to photo")
args = vars(ap.parse_args())

LABELS = {"001": '0', "002": '1', "003": '2', "004": '3', "005": '4', "006": '5', "007": '6', "008": '7', "009": '8',
          "010": '9', "011": 'A', "012": 'B', "013": 'C', "014": 'D', "015": 'E', "016": 'F', "017": 'G', "018": 'H',
          "019": 'I', "020": 'J', "021": 'K', "022": 'L',
          "023": 'M', "024": 'N', "025": 'O', "026": 'P', "027": 'Q', "028": 'R', "029": 'S', "030": 'T', "031": 'U',
          "032": 'V', "033": 'W', "034": 'X', "035": 'Y', "036": 'Z', "037": 'a', "038": 'b', "039": 'c', "040": 'd',
          "041": 'e', "042": 'f', "043": 'g', "044": 'h',
          "045": 'i', "046": 'j', "047": 'k', "048": 'l', "049": 'm', "050": 'n', "051": 'o', "052": 'p', "053": 'q',
          "054": 'r', "055": 's', "056": 't', "057": 'u', "058": 'v', "059": 'w', "060": 'x', "061": 'y', "062": 'z'}

imagePaths = list(paths.list_images(args["photo"]))
data = []
labels = []

for i in np.random.choice(np.arange(0, len(imagePaths)), size=(10,)):
    label = imagePaths[i].split(os.path.sep)[-2][-3:]
    image = cv2.imread(imagePaths[i])
    image = cv2.resize(image, (32, 32))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, dstCn=1)
    data.append(image)
    labels.append(LABELS[label])
allLabels = list(LABELS.values())

kerasData = np.divide(data, 255)
kerasData = kerasData[..., np.newaxis]
data = np.divide(data, 255)

lb = LabelBinarizer()
allLabels = lb.fit_transform(allLabels)

kerasModel = load_model('E:\Studia\magisterka\OCRNetwork\output/20_epok_1_channel_all_photo_32x32/lenet_model.h5')

tensorflowNet = cv2.dnn.readNetFromTensorflow('E:\Studia\magisterka\OCRNetwork\output/20_epok_1_channel_all_photo_32x32/tensorflow_model.pb')


print("Keras results:")
for i in range(len(kerasData)):
    kerasPredictions = kerasModel.predict(kerasData[np.newaxis, i])
    prediction = kerasPredictions.argmax(axis=1)
    prediction = lb.classes_[prediction]
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], labels[i]))

millis = int(round(time.time()*1000))
print("Tensorflow results:")
for i in range(len(data)):
    image = data[i].astype('float32')
    blob = cv2.dnn.blobFromImage(image, size=(32, 32), swapRB=True, crop=False)
    tensorflowNet.setInput(blob)
    cvOut = tensorflowNet.forward()
    prediction = cvOut.argmax(axis=1)
    prediction = lb.classes_[prediction]
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], labels[i]))
print("Estimated time: ", (int(round(time.time()*1000)) - millis), "ms")
