# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:01:35 2021

@author: krist
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import imutils

images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
images_right = glob.glob('data/imgs//withoutOcclusions/right/*.png')

map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")

map2x = np.loadtxt('data/map2x.csv', delimiter = "\t").astype("float32")
map2y = np.loadtxt('data/map2y.csv', delimiter = "\t").astype("float32")

Q = np.array([[1, 0, 0, -646.284], 
              [0, 1, 0, -384.277],
              [0, 0, 0, 703.981],
              [0, 0, 0.00833374, -0]])

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#read in 
img_left = cv2.imread(images_left[600])
img_right = cv2.imread(images_right[600])
    
#rectify
imgU1 = np.zeros(img_left.shape[:2], np.uint8)
imgU1 = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
imgU2 = np.zeros(img_right.shape[:2], np.uint8)
imgU2 = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR, imgU2, cv2.BORDER_CONSTANT, 0)


#stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
#Create Block matching object. 
win_size = 5
min_disp = 4
num_disp = 112 - min_disp

stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
 numDisparities = num_disp,
 blockSize = 5,
 uniquenessRatio = 10,
 speckleWindowSize = 150,
 speckleRange = 2,
 disp12MaxDiff = -1,
 P1 = 8*3*5**2,
 P2 =32*3*5**2)

disparity = stereo.compute(imgU1, imgU2)
#plt.imshow(cv2.cvtColor(imgU1, cv2.COLOR_BGR2RGB))
plt.imshow(disparity, cmap='gray')
plt.show()

#%% Havent gotten past this point yet because the disparity looks bad
for i in range(len(images_left)):
    
    #read in 
    img_left = cv2.imread(images_left[i])
    img_right = cv2.imread(images_right[i])
    
    #rectify
    imgU1 = np.zeros(img_left.shape[:2], np.uint8)
    imgU1 = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
    imgU2 = np.zeros(img_right.shape[:2], np.uint8)
    imgU2 = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR, imgU2, cv2.BORDER_CONSTANT, 0)
    
    
    
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