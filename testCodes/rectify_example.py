# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 11:26:08 2021

@author: krist
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import imutils
import open3d as o3d
import copy

#the unoccluded images
images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
images_right = glob.glob('data/imgs//withoutOcclusions/right/*.png')


#the maps for rectifying stuff
map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")

map2x = np.loadtxt('data/map2x.csv', delimiter = "\t").astype("float32")
map2y = np.loadtxt('data/map2y.csv', delimiter = "\t").astype("float32")

def readAndRectify(i):
    #read in 
    img_left = cv2.imread(images_left[i])
    img_right = cv2.imread(images_right[i])
    
    #rectify
    imgU1 = np.zeros(img_left.shape[:2], np.uint8)
    imgU1 = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
    imgU2 = np.zeros(img_right.shape[:2], np.uint8)
    imgU2 = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR, imgU2, cv2.BORDER_CONSTANT, 0)
    
    return imgU1, imgU2


for i in range(1, len(images_left)):
    
    #read in and rectify two images (U1 = left and U2 = right)
    imgU1, imgU2 = readAndRectify(i)
    
    #do whatever you want with the rectified images for example show them
    cv2.imshow("Video left", imgU1)
    cv2.imshow("Video right", imgU2)
    
    #press q to stop the video
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()