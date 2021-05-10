# -*- coding: utf-8 -*-
"""
Created on Sun May  9 14:56:46 2021

@author: krist
"""
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import imutils
import open3d as o3d
import copy

#the unoccluded images
#images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
#images_right = glob.glob('data/imgs//withoutOcclusions/right/*.png')

#the occluded images
images_left = glob.glob('data/imgs//withOcclusions/left/*.png')
images_right = glob.glob('data/imgs//withOcclusions/right/*.png')

map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")

map2x = np.loadtxt('data/map2x.csv', delimiter = "\t").astype("float32")
map2y = np.loadtxt('data/map2y.csv', delimiter = "\t").astype("float32")


#movementMask = cv2.imread("data/movementMask.jpg", 0)
movementMask = cv2.imread("data/movementMaskOccluded.jpg", 0)

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
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask[movementMask!=255]=0
    
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
        
    return feat1, feat2, pic, good_new, good_old, status

def to3D(grayU1, grayU2, goodNewPoint):
    stereo = cv2.StereoBM_create(numDisparities=208, blockSize=7)
    stereo.setMinDisparity(0)
    stereo.setUniquenessRatio(4)
    stereo.setTextureThreshold(253)
    stereo.setSpeckleRange(157)
    stereo.setSpeckleWindowSize(147)
    stereo.setDisp12MaxDiff(1)
    disparity = stereo.compute(grayU1, grayU2)
    
    disparity2 = cv2.normalize(disparity, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    vector = np.array([[[goodNewPoint[0],
                        goodNewPoint[1], 
                        disparity[int(goodNewPoint[1])][int(goodNewPoint[0])]]]])
    
    point_3D = cv2.perspectiveTransform(vector, Q)
    return point_3D, disparity2


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
    return X, P, u, F, H, R, I


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
    if state == 0:
        if w>0 and x+w > 1200 and x+w < 1280:
            feat1 = cv2.goodFeaturesToTrack(grayU1_prev[y:y+h-1, x:x+w-1], maxCorners=50, qualityLevel=0.04, minDistance=3)
            feat1 = feat1 + np.array([x,y]).astype('float32')
            state = 1
    
    elif state ==1:
        original_frame = grayU1
        feat1, feat2, pic, good_new, good_old, status = featureDetection(grayU1_prev, grayU1, feat1, x, y, w, h, "draw")
        temp = []
        for j in range(len(good_new)):
            if (abs(good_new[j][0]-good_old[j][0])>1 or abs(good_new[j][1]-good_old[j][1])>1):
                temp.append(good_new[j])
        feat1 = np.array(temp).reshape(-1,1,2)
        X, P, u, F, H, R, I = initializeKalman()
        
        goodPointIdx = np.argmax(good_new, axis=0)[0]
        state = 2
    
    elif state == 2:
        if x+w <1000:
            state = 3
        elif w > 0:
            feat1, feat2, pic, good_new, good_old, status = featureDetection(grayU1_prev, grayU1, feat1, x, y, w, h, "draw") 
            goodPointIdx = -1#np.sum(status[:goodPointIdx])
            point_3D, disparity2 = to3D(grayU1, grayU2, good_new[goodPointIdx])
            
            
            #if we dont have a valid disparity value don't do othe measuremnt update
            if (disparity2[int(good_new[goodPointIdx][1])][int(good_new[goodPointIdx][0])] != 0):
                
                Z = point_3D.reshape(3,1) 
            
                #drawing the measurement
                measurement_2D, _ = cv2.projectPoints(np.array([[Z[0][0], Z[1][0], Z[2][0]]]), np.zeros(3), np.array([0., 0., 0.]), mtx_left, np.array([0., 0., 0., 0.]))
                cv2.circle(pic, (int(measurement_2D[0][0][0]), int(measurement_2D[0][0][1])), 5, (0, 0, 255), -1)
                cv2.circle(pic, (int(good_new[goodPointIdx][0]), int(good_new[goodPointIdx][1])), 5, (255, 255, 255), -1)
                cv2.circle(disparity2, (int(good_new[goodPointIdx][0]), int(good_new[goodPointIdx][1])), 7, (255, 255, 255), 1)
                cv2.imshow("Disparity", disparity2)
                
                #print(H.dot(X)[0][0], Z[0][0])
                #print(H.dot(X)[1][0], Z[1][0])
                #print(H.dot(X)[2][0], Z[2][0])
                #print()
                X, P = update(X, P, Z, H, R)
            
            feat1 = good_new.reshape(-1,1,2)
            
        X, P = predict(X, P, F, u)
        
        #drawing the prediction
        point_2D, _ = cv2.projectPoints(np.array([[X[0][0], X[2][0], X[4][0]]]), np.zeros(3), np.array([0., 0., 0.]), mtx_left,  np.array([0., 0., 0., 0.]))
        cv2.circle(pic, (int(point_2D[0][0][0]), int(point_2D[0][0][1])), 5, (0, 0, 0), -1)
            
    elif state == 3:
        if w>0 and x+w < 900:
            feat1 = cv2.goodFeaturesToTrack(grayU1_prev[y:y+h-1, x:x+w-1], maxCorners=50, qualityLevel=0.04, minDistance=3)
            feat1 = feat1 + np.array([x,y]).astype('float32')
            X, P, u, F, H, R, I = initializeKalman()
            X = np.array([[0], 
                          [0], 
                          [0],
                          [0], 
                          [0], 
                          [0]])
            state = 4
    elif state == 4:
        if w > 0 and x+w <700:
            state = 0
        elif w > 0:
            feat1, feat2, pic, good_new, good_old, status = featureDetection(grayU1_prev, grayU1, feat1, x, y, w, h, "draw") 
            goodPointIdx = -1#np.sum(status[:goodPointIdx])
            point_3D, disparity2 = to3D(grayU1, grayU2, good_new[goodPointIdx])
            
            
            #if we dont have a valid disparity value don't do othe measuremnt update
            if (disparity2[int(good_new[goodPointIdx][1])][int(good_new[goodPointIdx][0])] != 0):
                
                Z = point_3D.reshape(3,1) 
            
                #drawing the measurement
                measurement_2D, _ = cv2.projectPoints(np.array([[Z[0][0], Z[1][0], Z[2][0]]]), np.zeros(3), np.array([0., 0., 0.]), mtx_left, np.array([0., 0., 0., 0.]))
                cv2.circle(pic, (int(measurement_2D[0][0][0]), int(measurement_2D[0][0][1])), 5, (0, 0, 255), -1)
                cv2.circle(pic, (int(good_new[goodPointIdx][0]), int(good_new[goodPointIdx][1])), 5, (255, 255, 255), -1)
                cv2.circle(disparity2, (int(good_new[goodPointIdx][0]), int(good_new[goodPointIdx][1])), 7, (255, 255, 255), 1)
                cv2.imshow("Disparity", disparity2)
                
                #print(H.dot(X)[0][0], Z[0][0])
                #print(H.dot(X)[1][0], Z[1][0])
                #print(H.dot(X)[2][0], Z[2][0])
                #print()
                X, P = update(X, P, Z, H, R)
            
            feat1 = good_new.reshape(-1,1,2)
        
        
        X, P = predict(X, P, F, u)
        
        #drawing the prediction
        point_2D, _ = cv2.projectPoints(np.array([[X[0][0], X[2][0], X[4][0]]]), np.zeros(3), np.array([0., 0., 0.]), mtx_left,  np.array([0., 0., 0., 0.]))
        cv2.circle(pic, (int(point_2D[0][0][0]), int(point_2D[0][0][1])), 5, (0, 0, 0), -1)
      
    #show result
    if c != []:
        cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Video", pic)
    
    imgU1_prev = imgU1
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()