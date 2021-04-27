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
    fgmask = fgbg.apply(imgU1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    x,y,w,h = 0,0,0,0

    #Draw only the biggest contour if its size is over threshold
    if len(cnts) != 0:
        c = max(cnts, key = cv2.contourArea)
        if cv2.contourArea(c) > 4000 and cv2.contourArea(c) < 35000 :
            (x, y, w, h) = cv2.boundingRect(c)
            #cv2.rectangle(imgU1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return x,y,w,h

def featureDetection(grayU1_prev, grayU1, feat1, x, y, w, h, draw=""):
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
    #do the flow calculation only if features were found
    if feat1 is not None:
        feat2, status, error = cv2.calcOpticalFlowPyrLK(grayU1_prev, grayU1, feat1, None, **lk_params)
        
        good_new = feat2[status==1]
        good_old = feat1[status==1]
        
        #drawing the found features 
        if draw == "draw":
            for i in range(len(good_old)):
                #Check if the absolute value of the difference of the coordinates in 2 pictures is more than 10 pixels
                if (abs(good_new[i][0]-good_old[i][0])>1 or abs(good_new[i][1]-good_old[i][1])>1):
                    cv2.line(pic, (int(good_old[i][0]), int(good_old[i][1])), (int(good_new[i][0]), int(good_new[i][1])), (0, 255, 0), 2)
                    cv2.circle(pic, (int(good_old[i][0]), int(good_old[i][1])), 5, (0, 255, 0), -1)
        
    return feat1, feat2, pic, good_new

def to3D(grayU1, grayU2):
    stereo = cv2.StereoBM_create(numDisparities=208, blockSize=7)
    stereo.setMinDisparity(0)
    stereo.setUniquenessRatio(4)
    stereo.setTextureThreshold(253)
    stereo.setSpeckleRange(157)
    stereo.setSpeckleWindowSize(147)
    stereo.setDisp12MaxDiff(1)
    disparity = stereo.compute(grayU1, grayU2)
    disparity2 = cv2.normalize(disparity, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    return points_3D, disparity2


state = 0

#Read and rectify the very first frame (left image)
img_left = cv2.imread(images_left[0])
imgU1_prev = np.zeros(img_left.shape[:2], np.uint8)
imgU1_prev = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1_prev, cv2.BORDER_CONSTANT, 0)

for i in range(1, len(images_left)):
    
    #read in and rectify two images (U1 = left and U2 = right)
    imgU1, imgU2 = readAndRectify()
    
    #convert all the 3 images to gray for goodfeatures and disparity
    grayU1 = cv2.cvtColor(imgU1, cv2.COLOR_BGR2GRAY)
    grayU2 = cv2.cvtColor(imgU2, cv2.COLOR_BGR2GRAY)
    grayU1_prev = cv2.cvtColor(imgU1_prev, cv2.COLOR_BGR2GRAY)
    
    pic = copy.deepcopy(imgU1)
    
    #motion detection on left image (returns the centre and width and height of the surrounding rect)
    x,y,w,h = motionDetection(imgU1)
    
    #detecting the first features at  the beginning of the treadmill
    if w>0 and x > 600 and x < 1100 and state == 0:
        feat1 = cv2.goodFeaturesToTrack(grayU1_prev[y:y+h-1, x:x+w-1], maxCorners=50, qualityLevel=0.04, minDistance=3)
        feat1 = feat1 + np.array([x,y]).astype('float32')
        state = 1
        
    elif state == 1:
        if w > 0 and x<600:
            state = 0
        elif w > 0:
            feat1, feat2, pic, good_new = featureDetection(grayU1_prev, grayU1, feat1, x, y, w, h, "draw") 
            points_3D, disparity2 = to3D(grayU1, grayU2)
            feat1 = good_new.reshape(-1,1,2)
        
        
    #show result
    cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Video", pic)
    imgU1_prev = imgU1
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()