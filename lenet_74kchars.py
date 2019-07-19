# set the matplotlib backend so figures can be saved in the background
import os

import matplotlib
from keras_preprocessing.image import ImageDataGenerator

matplotlib.use("Agg")

# import the necessary packages
from sklearn.model_selection import train_test_split
from lenet import LeNet
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-e", "--epochs", type=int, default=20,
                help="# of epochs to train our network for")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output loss/accuracy plot")
ap.add_argument("-s", "--save-model", type=int, default=-1,
                help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
                help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
                help="(optional) path to weights file")
args = vars(ap.parse_args())

LABELS = {"001": '0', "002": '1', "003": '2', "004": '3', "005": '4', "006": '5', "007": '6', "008": '7', "009": '8',
          "010": '9', "011": 'A', "012": 'B', "013": 'C', "014": 'D', "015": 'E', "016": 'F', "017": 'G', "018": 'H',
          "019": 'I', "020": 'J', "021": 'K', "022": 'L',
          "023": 'M', "024": 'N', "025": 'O', "026": 'P', "027": 'Q', "028": 'R', "029": 'S', "030": 'T', "031": 'U',
          "032": 'V', "033": 'W', "034": 'X', "035": 'Y', "036": 'Z', "037": 'a', "038": 'b', "039": 'c', "040": 'd',
          "041": 'e', "042": 'f', "043": 'g', "044": 'h',
          "045": 'i', "046": 'j', "047": 'k', "048": 'l', "049": 'm', "050": 'n', "051": 'o', "052": 'p', "053": 'q',
          "054": 'r', "055": 's', "056": 't', "057": 'u', "058": 'v', "059": 'w', "060": 'x', "061": 'y', "062": 'z'}

imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2][-3:]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    labels.append(LABELS[label])

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.2, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                         width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
#opt = SGD(lr=0.01)
opt = Adam(lr=1e-4, decay=1e-4 / args["epochs"])
model = LeNet.build(numChannels=3, width=64, height=64,
                    numClasses=62,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

print(model.summary())

# only train and evaluate the model if we *are not* loading a
# pre-existing model
if args["load_model"] < 0:
    print("[INFO] training...")
    # H = model.fit(trainX, trainY, batch_size=128, epochs=args["epochs"],
    #           verbose=1)
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX),  # // 32,
                            epochs=args["epochs"])

    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testX, testY,
                                      batch_size=64, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
    print("[INFO] loss: {:.2f}%".format(loss * 100))

    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=64)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=lb.classes_))

    # plot the training loss and accuracy
    N = args["epochs"]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

# check to see if the model should be saved to file
if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save("output/lenet_model.h5", include_optimizer=False, overwrite=True)
    model.save_weights(args["weights"], overwrite=True)
    with open("output/model_architecture.json", "w") as f:
        f.write(model.to_json())

# randomly select a few testing digits
for i in np.random.choice(np.arange(0, len(testY)), size=(10,)):
    # classify the digit
    probs = model.predict(testX[np.newaxis, i])
    prediction = probs.argmax(axis=1) #TODO: change predicted number to corresponding character
    prediction = lb.classes_[prediction]

    # extract the image from the testData if using "channels_first"
    # ordering
    if K.image_data_format() == "channels_first":
        image = (testX[i][0] * 255).astype("uint8")

    # otherwise we are using "channels_last" ordering
    else:
        image = (testX[i] * 255).astype("uint8")

    # merge the channels into one image
    # image = cv2.merge([image] * 3)

    # resize the image from a 28 x 28 image to a 96 x 96 image so we
    # can better see it
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)

    # show the image and prediction
    cv2.putText(image, str(prediction[0]), (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
                                                    lb.classes_[np.argmax(testY[i])]))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)
