# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 11:46:03 2021

@author: krist
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import imutils

images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))


assert images_left
for fname in images_left:
    #read in 
    img = cv2.imread(fname)
    
    #rectify
    imgU1 = np.zeros(img.shape[:2], np.uint8)
    imgU1 = cv2.remap(img, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
    
    fgmask = fgbg.apply(imgU1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #Draw only the biggest contour if its size is over threshold
    if len(cnts) != 0:
        c = max(cnts, key = cv2.contourArea)
        if cv2.contourArea(c) > 4000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(imgU1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Video", imgU1)
    cv2.imshow("Thresh", fgmask)
    
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()