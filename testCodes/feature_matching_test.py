# -*- coding: utf-8 -*-
"""
Created on Tue May  4 21:37:12 2021

@author: kuion
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
MIN_MATCH_COUNT = 3

#the unoccluded images
images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
images_right = glob.glob('data/imgs//withoutOcclusions/right/*.png')

img2 = cv2.imread(images_left[500])
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
x = 990
y = 270
w = 150
h = 140
#cv2.rectangle(img2, (x, y), (x + w, y + h), 255, 2)
img1 = img2[y:y+h, x:x+w].copy()

img2 = cv2.imread(images_left[620])
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#img1 = cv2.imread('box.png',0)          # queryImage
#img2 = cv2.imread('box_in_scene.png',0) # trainImage

# while 1:
#     cv2.imshow('beginning',img1)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break
# cv2.destroyAllWindows()
#%%

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
#%%

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
import copy

#the unoccluded images
images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
images_right = glob.glob('data/imgs//withoutOcclusions/right/*.png')

#the occluded images
#images_left = glob.glob('data/imgs//withOcclusions/left/*.png')
#images_right = glob.glob('data/imgs//withOcclusions/right/*.png')

map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")

map2x = np.loadtxt('data/map2x.csv', delimiter = "\t").astype("float32")
map2y = np.loadtxt('data/map2y.csv', delimiter = "\t").astype("float32")


mtx_left = np.array([[705.127,	0,	621.042],
                     [0,	705.055,	370.571],
                     [0,	0,	1]])


Q = np.array([[1, 0, 0, -646.284], 
              [0, 1, 0, -384.277],
              [0, 0, 0, 703.981],
              [0, 0, 0.00833374, -0]])

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

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

def motionDetection(img):
    fgmask = fgbg.apply(imgU1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    x,y,w,h,c = 0,0,0,0,[]

    #Draw only the biggest contour if its size is over threshold
    if len(cnts) != 0:
        cnt = max(cnts, key = cv2.contourArea)
        if cv2.contourArea(cnt) > 4000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            c = [cnt]
    return x,y,w,h,c

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
                    cv2.circle(pic, (int(good_new[i][0]), int(good_new[i][1])), 5, (0, 255, 0), -1)
        
    return feat1, feat2, pic, good_new


state = 0

#Read and rectify the very first frame (left image)
img_left = cv2.imread(images_left[0])
imgU1_prev = np.zeros(img_left.shape[:2], np.uint8)
imgU1_prev = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1_prev, cv2.BORDER_CONSTANT, 0)

for i in range(1, len(images_left)):
    
    #read in and rectify two images (U1 = left and U2 = right)
    imgU1, imgU2 = readAndRectify(i)
    
    #convert all the 3 images to gray for goodfeatures and disparity
    grayU1 = cv2.cvtColor(imgU1, cv2.COLOR_BGR2GRAY)
    grayU2 = cv2.cvtColor(imgU2, cv2.COLOR_BGR2GRAY)
    grayU1_prev = cv2.cvtColor(imgU1_prev, cv2.COLOR_BGR2GRAY)
    
    pic = copy.deepcopy(imgU1)
    
    #motion detection on left image (returns the centre and width and height of the surrounding rect)
    x,y,w,h,c = motionDetection(imgU1)
    
    #detecting the first features at  the beginning of the treadmill
    if w>0 and x+w > 700 and x+w < 1230 and state == 0:
        imgItem = grayU1[y:y+h, x:x+w].copy()
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(imgItem,None)
        state = 1
    
    #search for features
    elif state == 1:
        if w > 0 and x+w <700:
            state = 0
        elif w > 0:
            imgItem2 = grayU1[y:y+h, x:x+w].copy()
            kp2, des2 = sift.detectAndCompute(imgItem2,None)
            # BFMatcher with default params
            bf = cv2.BFMatcher()
            # Finds k best matches
            matches = bf.knnMatch(des1,des2,k=2)
            # Apply ratio test, checks if first neighbour is significantly closer than the second one
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            # Pick only 10 best out of those
            good = sorted(good, key = lambda x:x[0].distance)[:10]
            # cv.drawMatchesKnn expects list of lists as matches.
            grayU1 = cv2.drawMatchesKnn(imgItem,kp1,imgItem2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # Get list of good match indeces of keypoints in second(/trainIdx) image; queryIdx is the first one
            goodi = [x[0].trainIdx for x in good]
            for i in goodi:
                cv2.circle(pic, (int(kp2[i].pt[0]+x), int(kp2[i].pt[1]+y)), 5, (0, 255, 0), -1)
            
      
    #show result
    if c != []:
        cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Video", pic)
    cv2.imshow("FindFeat", grayU1)
    
    imgU1_prev = imgU1
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()