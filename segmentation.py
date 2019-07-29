import cv2
import numpy as np
import time

from imutils import paths

imagePaths = list(paths.list_images('E:/Studia/magisterka/fotki_do_testowania_segmentacji/'))

data = []
hist = []
horizontal = []
processedSections = []
finalCharacters = []
th = 2

for i in range(len(imagePaths)):
    image = cv2.imread(imagePaths[i])
    data.append(image)

for i in range(len(data)):
    millis = int(round(time.time() * 1000))
    # convert into grayscale
    data[i] = cv2.cvtColor(data[i], cv2.COLOR_RGB2GRAY, dstCn=1)
    # create histogram
    hist.append(cv2.calcHist([data[i]], [0], None, [256], [0, 256]))
    # take maximum value from histogram
    maxValue = np.argmax(hist[i])
    # if maxValue is not high enough then text is in light colour, so invert it
    if maxValue < 180:
        data[i] = cv2.bitwise_not(data[i])
    # use blur to get better results in binarization
    data[i] = cv2.GaussianBlur(data[i], (5, 5), 0)
    temp, data[i] = cv2.threshold(data[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # find horizontal lines on image to separate lines with text
    horizontal.append(cv2.reduce(cv2.bitwise_not(data[i]), 1, cv2.REDUCE_AVG).reshape(-1))
    # find upper and lower lines for each text segment
    H, W = data[i].shape[:2]
    uppers = [y for y in range(H - 1) if horizontal[i][y] <= th < horizontal[i][y + 1]]
    lowers = [y for y in range(H - 1) if horizontal[i][y] > th >= horizontal[i][y + 1]]
    # add safety lines
    if len(lowers) > len(uppers):
        uppers.insert(0, 0)
    if len(lowers) < len(uppers):
        lowers.append(H)

    # process the text segments to new images
    segments = []
    # separate segments to words
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    wordsInSegment = []
    for j in range(len(uppers)):
        segments.append(data[i][uppers[j]:lowers[j], 0:W])
        segments[j] = cv2.copyMakeBorder(segments[j], 10, 10, 0, 0, cv2.BORDER_CONSTANT, value=255)

        dilated = cv2.erode(segments[j], kernel)
        # cv2.imshow("", dilated)
        # cv2.waitKey(0)
        contours = cv2.findContours(cv2.bitwise_not(dilated), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(contours, key=lambda c: min(min(c[:, :, 0])))
        words = []
        dilateKernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (3, 3))
        # separate words to characters
        finalCharacters = []
        for k in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[k])
            words.append(segments[j][y:y+h, x:x+w])

            eroded = cv2.dilate(words[k], dilateKernel)
            # cv2.imshow("", eroded)
            # cv2.waitKey(0)
            charContours = cv2.findContours(cv2.bitwise_not(eroded), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            charContours = sorted(charContours, key=lambda c: min(min(c[:, :, 0])))
            characters = []
            for l in range(len(charContours)):
                x, y, w, h = cv2.boundingRect(charContours[l])
                height, width = words[k].shape
                if width > x+w > 50 and height/2 < y+h < height:
                    characters.append(words[k][y:y+h, x:x+w])

            finalCharacters.append(characters)
        wordsInSegment.append(finalCharacters)
    processedSections.append(wordsInSegment)
    print("Estimated time: ", (int(round(time.time() * 1000)) - millis), "ms")
    # TODO: segmentation of single characters

# show results
for i in range(len(processedSections)):
    wordsInSegment = processedSections[i]
    for j in range(len(wordsInSegment)):
        words = wordsInSegment[j]
        for k in range(len(words)):
            characters = words[k]
            for l in range(len(characters)):
                cv2.imshow("", characters[l])
                cv2.waitKey(0)
