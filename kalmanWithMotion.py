# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:32:12 2021

@author: krist
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


#images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
#images_right = glob.glob('data/imgs//withoutOcclusions/right/*.png')

images_left = glob.glob('data/imgs//withOcclusions/left/*.png')
images_right = glob.glob('data/imgs//withOcclusions/right/*.png')

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
    x,y,w,h,c = 0,0,0,0,[]

    #Draw only the biggest contour if its size is over threshold
    if len(cnts) != 0:
        cnt = max(cnts, key = cv2.contourArea)
        if cv2.contourArea(cnt) > 4000 and cv2.contourArea(cnt) < 50000 :
            (x, y, w, h) = cv2.boundingRect(cnt)
            c = [cnt]
    return x,y,w,h,c


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


def update(x, P, Z, H, R):
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

### Initialize Kalman filter ###
# The initial state (6x1).
X = np.array([[0], #x position
              [0], #x velocity
              [0], #y position
              [0], #y velocity
              [0], #z position
              [0]]) #z velocity

# The initial uncertainty (6x6).
P = np.array([[1000000, 0, 0, 0, 0, 0],
              [0, 1000000, 0, 0, 0, 0],
              [0, 0, 1000000, 0, 0, 0],
              [0, 0, 0, 1000000, 0, 0],
              [0, 0, 0, 0, 1000000, 0],
              [0, 0, 0, 0, 0, 1000000]])

# The external motion (6x1).
u = np.array([[0], 
              [0], 
              [0],
              [0], 
              [0], 
              [0]])

# The transition matrix (6x6). 
F = np.array([[1, 1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1]])

# The observation matrix (2x6).
H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0]])

# The measurement uncertainty.
R = 1

I = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])


state = 0

for i in range(1, len(images_left)):
    
    #read in and rectify two images (U1 = left and U2 = right)
    imgU1, imgU2 = readAndRectify()
    
    #convert all the 3 images to gray for goodfeatures and disparity
    grayU1 = cv2.cvtColor(imgU1, cv2.COLOR_BGR2GRAY)
    grayU2 = cv2.cvtColor(imgU2, cv2.COLOR_BGR2GRAY)
    
    pic = copy.deepcopy(imgU1)
    
    #motion detection on left image (returns the centre and width and height of the surrounding rect)
    x,y,w,h,c = motionDetection(imgU1)
    
    
    #new object has been placed at the beginning of the treadmill
    if w>0 and x > 600 and x < 1100 and state == 0:
        state = 1
        X = np.array([[0], #x position
              [0], #x velocity
              [0], #y position
              [0], #y velocity
              [0], #z position
              [0]]) #z velocity

        # The initial uncertainty (6x6).
        P = np.array([[1000000, 0, 0, 0, 0, 0],
              [0, 1000000, 0, 0, 0, 0],
              [0, 0, 1000000, 0, 0, 0],
              [0, 0, 0, 1000000, 0, 0],
              [0, 0, 0, 0, 1000000, 0],
              [0, 0, 0, 0, 0, 1000000]])
      
    #tracking
    elif state == 1:
        
        #the object reached the end of the treadmill start waiting for the new one
        if w > 0 and x<600:
            state = 0
            
        #if rectangle is found get the measurement for Kalman and do update and predict
        #if not found do only predict
        elif w> 0:
            points_3D, disparity2 = to3D(grayU1, grayU2)
            cnt_mask = np.zeros_like(grayU1)
            cv2.drawContours(cnt_mask, c, 0, 255, -1)
            object_coordinates = points_3D[(cnt_mask == 255)]
            
            #removing infinite values
            coordinate_mask = np.isfinite(object_coordinates).any(axis=1)
            object_coordinates = object_coordinates[coordinate_mask] 
            
            #finding the mean (centre of the object in 3D)
            measurement = np.mean(object_coordinates, axis = 0)
            Z = np.array([measurement]).T
            
            #drawing the measurement
            measurement_2D, _ = cv2.projectPoints(np.array([[Z[0][0], Z[1][0], Z[2][0]]]), np.zeros(3), np.array([0., 0., 0.]), mtx_left, np.array([0., 0., 0., 0.]))
            cv2.circle(pic, (int(measurement_2D[0][0][0]), int(measurement_2D[0][0][1])), 5, (0, 0, 255), -1)
            
            X, P = update(X, P, Z, H, R)
    
        
        X, P = predict(X, P, F, u)
        
        #drawing the prediction
        point_2D, _ = cv2.projectPoints(np.array([[X[0][0], X[2][0], X[4][0]]]), np.zeros(3), np.array([0., 0., 0.]), mtx_left,  np.array([0., 0., 0., 0.]))
        cv2.circle(pic, (int(point_2D[0][0][0]), int(point_2D[0][0][1])), 5, (255, 0, 0), -1)
        
    #show result
    cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Video", pic)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()