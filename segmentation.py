import cv2
import os
import numpy as np
import time

from imutils import paths
from matplotlib import pyplot as plt

imagePaths = list(paths.list_images('E:/Studia/magisterka/fotki_do_testowania_segmentacji/'))

data = []
hist = []
horizontal = []
processedSections = []
th = 2

for i in range(len(imagePaths)):
    image = cv2.imread(imagePaths[i])
    data.append(image)

for i in range(len(data)):
    millis = int(round(time.time() * 1000))
    data[i] = cv2.cvtColor(data[i], cv2.COLOR_RGB2GRAY, dstCn=1)
    hist.append(cv2.calcHist([data[i]], [0], None, [256], [0, 256]))
    maxValue = np.argmax(hist[i])
    if maxValue < 180:
        data[i] = cv2.bitwise_not(data[i])
    data[i] = cv2.GaussianBlur(data[i], (5, 5), 0)
    temp, data[i] = cv2.threshold(data[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # data[i] = cv2.adaptiveThreshold(data[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    horizontal.append(cv2.reduce(cv2.bitwise_not(data[i]), 1, cv2.REDUCE_AVG).reshape(-1))
    H, W = data[i].shape[:2]
    uppers = [y for y in range(H - 1) if horizontal[i][y] <= th < horizontal[i][y + 1]]
    lowers = [y for y in range(H - 1) if horizontal[i][y] > th >= horizontal[i][y + 1]]
    if len(lowers) > len(uppers):
        uppers.insert(0, 0)
    if len(lowers) < len(uppers):
        lowers.append(H)

    for j in range(len(uppers)):
        processedSections.append(data[i][uppers[j]:lowers[j], 0:W])
    print("Estimated time: ", (int(round(time.time() * 1000)) - millis), "ms")

    # cv2.imshow("Binarization", data[i])

for i in range(len(processedSections)):
    cv2.imshow("", processedSections[i])
    cv2.waitKey(0)
