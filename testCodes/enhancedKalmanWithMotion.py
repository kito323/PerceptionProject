# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:32:12 2021

@author: krist
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import imutils
import copy
import os 

from joblib import load
from classification.utilsBOVW import *
from classification.testBOVW import predictLabel

os.environ['DISPLAY'] = ':0'

#from rectification import mtx_left, rvecs_left, tvecs_left, dist_left
dist_left = np.array([-0.32948,	0.141779,	-0.000115869,	0.000253564,	-0.0310092])

mtx_left = np.array([[705.127,	0,	621.042],
                    [0,	705.055,	370.571],
                    [0,	0,	1]])

 # LOAD CLASSIFICATOR
svm, kmeans, scaler, num_cluster, imgs_features = load('classification/BOVW_300_copy.pkl')
# Create sift object
sift = cv2.xfeatures2d.SIFT_create()

#images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
#images_right = glob.glob('data/imgs//withoutOcclusions/right/*.png')

images_left = glob.glob('data/imgs//withOcclusions/left/*.png')
images_right = glob.glob('data/imgs//withOcclusions/right/*.png')
images_left.sort()
images_right.sort()
assert images_right, images_left
assert (len(images_right) == len(images_left))

map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")

map2x = np.loadtxt('data/map2x.csv', delimiter = "\t").astype("float32")
map2y = np.loadtxt('data/map2y.csv', delimiter = "\t").astype("float32")

#movementMask = cv2.imread("data/movementMask.jpg", 0)
movementMask = cv2.imread("data/movementMaskOccluded.jpg", 0)

Q = np.array([[1, 0, 0, -646.284], 
            [0, 1, 0, -384.277],
            [0, 0, 0, 703.981],
            [0, 0, 0.00833374, -0]])

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False, history = 600, varThreshold = 20)
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
    label = ['None']
    fgmask = fgbg.apply(img)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask[movementMask!=255]=0
    
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    x,y,w,h,c = 0,0,0,0,[]

    #Draw only the biggest contour if its size is over threshold
    if len(cnts) != 0:
        cnt = max(cnts, key = cv2.contourArea)
        if cv2.contourArea(cnt) > 3500 and cv2.contourArea(cnt) < 50000:
            (x, y, w, h) = cv2.boundingRect(cnt)
            c = [cnt]
            rectCenter = calculate_rect_center(x,y,w,h)
            # print(rectCenter)
            # Predict when the object has passed the occlusion
            if rectCenter[0] > 390 and rectCenter[0] < 700:
                cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img = pic[y:y+h, x:x+w]
                label = predictLabel(img, sift, num_cluster, kmeans, svm, scaler, imgs_features)
            # Preict when the object before the occlusion
            elif rectCenter[0] > 1122 and 300 < rectCenter[1] and rectCenter[1] < 440:
                cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img = pic[y:y+h, x:x+w]
                label = predictLabel(img, sift, num_cluster, kmeans, svm, scaler, imgs_features)


    return x,y,w,h,c,fgmask,label[0]

def calculate_rect_center(x,y,w,h):
    c_x = int(x + w/2)
    c_y = int(y + h/2)
    return (c_x,c_y)

def getDisparityMap(grayU1, grayU2):
    stereo = cv2.StereoBM_create(numDisparities=208, blockSize=7) #208 and 7
    stereo.setMinDisparity(0)
    stereo.setUniquenessRatio(4)
    stereo.setTextureThreshold(253)
    stereo.setSpeckleRange(157)
    stereo.setSpeckleWindowSize(147)
    stereo.setDisp12MaxDiff(1)
    disparity = stereo.compute(grayU1, grayU2)
    disparity2 = cv2.normalize(disparity, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity, disparity2


def update(x, P, Z, H, R, I):
    Y = Z-(np.dot(H,x))
    S =  (np.linalg.multi_dot([H, P, np.transpose(H)]))+R
    K = np.linalg.multi_dot([P, np.transpose(H), np.linalg.pinv(S)])
    X_next = x+np.dot(K,Y)
    P_next = np.dot((I-np.dot(K,H)),P)
    return X_next, P_next
    

def predict(x, P, F, u):
    X_next = (np.dot(F,x))+u
    P_next = np.linalg.multi_dot([F, P, np.transpose(F)])
    return X_next, P_next

def initializeKalman():
    ### Initialize Kalman filter ###
    
    # The initial state (6x1).
    X = np.array([[37.37],  #x position
                  [0],      #x velocity
                  [-5.449], #y position
                  [0],      #y velocity
                  [52.338], #z position
                  [0]])     #z velocity
                         
    
    # The initial uncertainty (6x6).
    P = np.identity(6)*10
    
    # The external motion (6x1).
    u = np.zeros((6, 1))
    
    # The transition matrix (6x6). 
    F = np.array([[1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1]])
    
    # The observation matrix (3x6).
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0]])
    
    # The measurement uncertainty.
    R = 1
    
    I = np.identity(6)
    
    return X, P, u, F, H, R, I

state = 0
frameCount = 0
log = []

for i in range(0, len(images_left)):
    
    #read in and rectify two images (U1 = left and U2 = right)
    imgU1, imgU2 = readAndRectify()
    
    #convert all the 3 images to gray for goodfeatures and disparity
    grayU1 = cv2.cvtColor(imgU1, cv2.COLOR_BGR2GRAY)
    grayU2 = cv2.cvtColor(imgU2, cv2.COLOR_BGR2GRAY)
    
    pic = copy.deepcopy(imgU1)
    
    #motion detection on left image (returns the centre and width and height of the surrounding rect)
    x,y,w,h,c,fgmask, label = motionDetection(imgU1)
    
    
    #waiting for object to reach the detection area 
    if state == 0:
        if w > 0 and x + w > 1200 and x + w < 1280: #change first to 1150 for unoccluded video
            X, P, u, F, H, R, I = initializeKalman()
            sublog = []
            frameCount = 0
            state = 1
    
    #tracking
    elif state == 1:
        
        #the right edge of the object reached a certain point -> start waiting for the new object
        if w > 0 and x + w < 700:
            state = 0

            
        #if motion is found get the measurement for Kalman and do update and predict
        elif w> 0 and (frameCount < 15 or x < 600):
            disparity, disparity2 = getDisparityMap(grayU1, grayU2)
            cnt_mask = np.zeros_like(grayU1)
            cv2.drawContours(cnt_mask, c, 0, 255, -1)
            
            #find the image x and y of the centre of the object 
            whiteCoordinates = np.argwhere(cnt_mask==255)
            centreOfWhite = np.mean(whiteCoordinates, axis = 0)          
            
            #remove points with below 0 disparity value and find the disparity of the centre of the object
            cnt_mask[disparity <= 0] = 0
            pointDisp = np.median(disparity[(cnt_mask == 255)])
            
            
            #put together the point x, y and disp and convert to 3D to get the measurement
            point = np.array([[[centreOfWhite[1], centreOfWhite[0], pointDisp]]])
            measurement = cv2.perspectiveTransform(point, Q)
            Z = measurement.reshape(3,1)
            
            sublog.append(Z.T)
            
            #drawing the measurement
            measurement_2D, _ = cv2.projectPoints(np.array([[Z[0][0], Z[1][0], Z[2][0]]]), np.zeros(3), np.array([0., 0., 0.]), mtx_left, np.array([0., 0., 0., 0.]))
            cv2.circle(pic, (int(measurement_2D[0][0][0]), int(measurement_2D[0][0][1])), 5, (0, 0, 255), -1)
            cv2.putText(pic, str(round(Z[0][0])) + " " + str(round(Z[1][0]))+ " " + str(round(Z[2][0])), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2) 
            
            X, P = update(X, P, Z, H, R, I)
            #cv2.imshow("Disparity", disparity2)
    
        #if motion not found do only predict
        X, P = predict(X, P, F, u)
        
        #drawing the prediction
        point_2D, _ = cv2.projectPoints(np.array([[H.dot(X)[0][0], (H.dot(X))[1][0], (H.dot(X))[2][0]]]), np.zeros(3), np.array([0., 0., 0.]), mtx_left,  np.array([0., 0., 0., 0.]))
        cv2.circle(pic, (int(point_2D[0][0][0]), int(point_2D[0][0][1])), 5, (255, 0, 0), -1)
        cv2.putText(pic, str(round(X[0][0])) + " " + str(round(X[2][0])) + " " + str(round(X[4][0])), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2) 
        
        frameCount += 1
    #show result
    cv2.putText(pic, 'Object detected: ' + label, (10, 75), cv2.FONT_ITALIC, 0.75, (0,0,0), 2)
    
    cv2.imshow("Video", pic)
    cv2.imshow("Motion", fgmask)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()