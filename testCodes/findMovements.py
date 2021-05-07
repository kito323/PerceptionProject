# -*- coding: utf-8 -*-
"""
Created on Fri May  7 14:48:58 2021

@author: kuion
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 11:27:24 2021

@author: krist
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import imutils
import open3d as o3d
import copy
#from rectification import mtx_left, rvecs_left, tvecs_left, dist_left

dist_left = np.array([-0.32948,	0.141779,	-0.000115869,	0.000253564,	-0.0310092])

mtx_left = np.array([[705.127,	0,	621.042],
                     [0,	705.055,	370.571],
                     [0,	0,	1]])


images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
images_right = glob.glob('data/imgs//withoutOcclusions/right/*.png')

# images_left = glob.glob('data/imgs//withOcclusions/left/*.png')
# images_right = glob.glob('data/imgs//withOcclusions/right/*.png')

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

def readAndRectify():
    #read in 
    img_left = cv2.imread(images_left[i])
    img_right = cv2.imread(images_right[i])
    
    #rectify
    imgU1 = np.zeros(img_left.shape[:2], np.uint8)
    imgU1 = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
    imgU2 = np.zeros(img_right.shape[:2], np.uint8)
    imgU2 = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR, imgU2, cv2.BORDER_CONSTANT, 0)
    
    return imgU1, imgU2

def motionDetection(img):
    fgmask = fgbg.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    x,y,w,h,c = 0,0,0,0,[]

    #Draw only the biggest contour if its size is over threshold
    if len(cnts) != 0:
        cnt = max(cnts, key = cv2.contourArea)
        if cv2.contourArea(cnt) > 4000 and cv2.contourArea(cnt) < 50000 :
            (x, y, w, h) = cv2.boundingRect(cnt)
            c = [cnt]
    return x,y,w,h,c


state = 0
firstImg = cv2.imread(images_left[0], 0)
addedMask = np.zeros_like(firstImg)
addedMask = addedMask.astype('float64')

for i in range(1, len(images_left)):
    
    #read in and rectify two images (U1 = left and U2 = right)
    imgU1, imgU2 = readAndRectify()
    
    #convert all the 3 images to gray for goodfeatures and disparity
    grayU1 = cv2.cvtColor(imgU1, cv2.COLOR_BGR2GRAY)
    grayU2 = cv2.cvtColor(imgU2, cv2.COLOR_BGR2GRAY)
    
    pic = copy.deepcopy(imgU1)
    
    #motion detection on left image (returns the centre and width and height of the surrounding rect)
    x,y,w,h,c = motionDetection(imgU1)
    
    cnt_mask = np.zeros_like(grayU1)
    cv2.drawContours(cnt_mask, c, 0, 255, -1)
    addedMask = addedMask + cnt_mask
        
    # #show result
    # cv2.imshow("Video", cnt_mask)
    
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord("q"):
    #     break
    
addedMask = addedMask/255
plt.imshow(addedMask, cmap='gray')
cv2.imwrite('movementImg.jpg', addedMask)
