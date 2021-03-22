from base64 import decode
import numpy as np
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pyzbar import pyzbar
import imutils
import cv2
import imutils
from PIL import Image
'''explaination
https://www.youtube.com/watch?v=3hwNXsn5Xns

'''

def decodeandDraw(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (11, 11))
    (_, thresh) = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
    # Perform canny edge detection
    canny = cv2.Canny(thresh, 120, 255, 1)
    structureElement = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 22))
    improvedCanny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, structureElement)  # closing to fill inside parts
    # ####################################### Improve for object #############################
    # construct a closing kernel and apply it to the threshold's image
    structureElement2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(improvedCanny, cv2.MORPH_CLOSE, structureElement2)
    # perform a series of erosions and dilations to remove noise
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.medianBlur(closed, 21)
    closed = cv2.dilate(closed, None, iterations=14)
    # find the barcodes in the image and decode each of the barcodes
    barcodes = pyzbar.decode(image)
    # loop over the detected barcodes
    for barcode in barcodes:
        # extract the bounding box location of the barcode and draw the
        # bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        # the barcode data is a bytes object so if we want to draw it on
        # our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # draw the barcode data and barcode type on the image
        # text = "{} ({})".format(barcodeData, barcodeType)
        # cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
        # print the barcode type and data to the terminal
        print("BarCode OR QR?\n Found {} ,data inside: {}".format(barcodeType, barcodeData))
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)



# Algorithm
# Compute the Scharr gradient magnitude representations in both the x and y direction.
# Subtract the y-gradient from the x-gradient to reveal the barcoded region.
# Blur and threshold the image.
# use Canny Edge detection
# Apply a closing kernel to the Edge detection image.
# Perform a series of dilation and erosion.
# Find the largest contour in the image, which is now presumably the barcode.
def getAllQRCode(image):
    qrCodeList = []
    # Gaussian Blur
    # Gaussian = cv2.GaussianBlur(image, (7, 7), 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    # print("Y {}".format(gradY))
    # print("X {}".format(gradX))
    # print("depth {}".format(ddepth))
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (11, 11))

    plt.title("Blurred image")
    plt.axis("off")
    plt.imshow(blurred)
    plt.show()

    (_, thresh) = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)

    plt.title("thresh")
    plt.axis("off")
    plt.imshow(thresh)
    plt.show()

    # Perform canny edge detection, parameter: threshold1 threshold2
    canny = cv2.Canny(thresh, 120, 255, 1)
    plt.title("canny")
    plt.axis("off")
    plt.imshow(canny)
    plt.show()

    structureElement = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 22))
    improvedCanny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, structureElement) # closing to fill inside parts
    plt.title("Improve canny")
    plt.axis("off")
    plt.imshow(improvedCanny)
    plt.show()
    # ####################################### Improve for object #############################
    # construct a closing kernel and apply it to the threshold's image
    structureElement2= cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(improvedCanny, cv2.MORPH_CLOSE, structureElement2)

    # perform a series of erosions and dilations to remove noise
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.medianBlur(closed, 21)
    closed = cv2.dilate(closed, None, iterations=14)

    plt.title("closed")
    plt.axis("off")
    plt.imshow(closed)
    plt.show()

    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # draw a bounding box arounded the detected barcode and display the
    # image
    original = image.copy()
    image_number = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
        ROI = original[y:y + h, x:x + w]
        # decode QR to string
        barcodes = pyzbar.decode(image)
        barcodeData = barcodes[image_number].data.decode("utf-8")
        barcodeType = barcodes[image_number].type
        print("BarCode OR QR?\n Found {} ,data inside: {}".format(barcodeType, barcodeData))

        qrCodeList.append(ROI)
        cv2.imwrite("ROI_{}.png".format(image_number), ROI)  # ROI image to write
        image_number += 1

    # draw image with contour Area
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    return qrCodeList


# test function
image = cv2.imread("6.1.bmp")
getAllQRCode(image)
# decodeandDraw(image)
# for i in range(len(temp)):
#     plt.title("out{0}".format(i))
#     plt.axis("off")
#     plt.imshow(temp[i])
#     plt.show()
