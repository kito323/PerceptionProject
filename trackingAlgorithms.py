# -*- coding: utf-8 -*-
"""
Created on Thu May  6 09:40:30 2021

@author: krist
"""

import cv2
import sys
import numpy as np
import glob
import copy
import imutils

#Using the following tutorial https://learnopencv.com/object-tracking-using-opencv-cpp-python/

images_left = glob.glob('data/imgs//withOcclusions/left/*.png')
images_right = glob.glob('data/imgs//withOcclusions/right/*.png')

map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")

map2x = np.loadtxt('data/map2x.csv', delimiter = "\t").astype("float32")
map2y = np.loadtxt('data/map2y.csv', delimiter = "\t").astype("float32")

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

def motionDetection(img):
    fgmask = fgbg.apply(img)
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
    return x,y,w,h,c,fgmask



if __name__ == '__main__' :

    state = 0
    
    # Set up tracker.
    tracker = cv2.TrackerTLD_create()
    tracker_type = "TLD"

    # Uncomment the line below to select a bounding box by hand
    #bbox = cv2.selectROI(frame, False)

    for i in range(0, len(images_left)):
        
        #read in and rectify a new frame 
        img_left = cv2.imread(images_left[i])
        imgU1 = np.zeros(img_left.shape[:2], np.uint8)
        frame = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
        
        #make a copy for drawing
        pic = copy.deepcopy(frame)
        
        #detecting motion and finding the bounding rectangle
        x,y,w,h,c,fgmask = motionDetection(frame)
        
        # Start timer
        timer = cv2.getTickCount()
        
        #state for waiting for the object to arrive at the detection spot
        if state == 0:
            if w>0 and x+w > 1200 and x+w < 1250:
                #set the box as the bounding box for the tracker and initialize it
                bbox = (x,y,w,h)
                ok = tracker.init(frame, bbox)
                state = 1
            else:
                pass
        
        #The state for tracking
        elif state == 1:
            if w > 0 and x+w<600:
                state = 0
                
            #Do the tracking only when there is motion
            if w != 0:
                # Update tracker with the new frame
                ok, bbox = tracker.update(frame)
    
                # Draw bounding box if found otherwise failure text
                if ok:
                    # Tracking success
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(pic, p1, p2, (255,0,0), 2, 1)
                else :
                    # Tracking failure
                    cv2.putText(pic, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                    
            else:
                pass
            
        # Display tracker type on frame
        cv2.putText(pic, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # Display FPS on frame
        cv2.putText(pic, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        cv2.imshow("Tracking", pic)
        cv2.imshow("Mask", fgmask)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == ord("q") : break
    
cv2.destroyAllWindows()