# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:27:46 2021

@author: krist
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")


imgU2 = None

assert images_left
for fname in images_left:
    
    img = cv2.imread(fname)
    
    imgU1 = np.zeros(img.shape[:2], np.uint8)
    imgU1 = cv2.remap(img, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)

    if imgU2 is None:
        imgU2 = imgU1
        continue
    
    gray1 = cv2.cvtColor(imgU1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(imgU2, cv2.COLOR_BGR2GRAY)

    feat1 = cv2.goodFeaturesToTrack(gray1, maxCorners=200, qualityLevel=0.01, minDistance=5)
    feat2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, feat1, None)

    pic=imgU2.copy()
    for i in range(len(feat1)):
        #Check if the absolute value of the difference of the coordinates in 2 pictures is more than 10 pixels
        if (abs(feat2[i][0][0]-feat1[i][0][0])>10 or abs(feat2[i][0][1]-feat1[i][0][1])>1):
            cv2.line(pic, (feat1[i][0][0], feat1[i][0][1]), (feat2[i][0][0], feat2[i][0][1]), (0, 255, 0), 2)
            cv2.circle(pic, (feat1[i][0][0], feat1[i][0][1]), 5, (0, 255, 0), -1)

    
    cv2.imshow("Video", pic)
    imgU2 = imgU1
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()