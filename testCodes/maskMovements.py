# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:57:37 2021

@author: kuion
"""
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
import open3d as o3d

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

#Functions for making tracbars for thresholding
def empty_function(*args):
    pass

def Trackbar(gray1, background):
    win_name = "ThresholdTrackbar"
    
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow(win_name, 900,400) #1000, 500
    
    cv2.createTrackbar("thresh", win_name, 127, 255, empty_function)
    
    while True:
        thresh = cv2.getTrackbarPos("thresh", win_name)
        
        ret,thresh1 = cv2.threshold(gray1,thresh,255,cv2.THRESH_BINARY)
        
        res = cv2.bitwise_and(background, background, mask=thresh1)
        
        cv2.imshow(win_name, res)

        # Code exits "while true loop" by pressing letter 'c'
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return thresh1, thresh


img = cv2.imread('data/movementImg.jpg', 0)

images_left = glob.glob('data/imgs/withoutOcclusions/left/*.png')
img_left = cv2.imread(images_left[0])

map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")

#rectify
imgU1 = np.zeros(img_left.shape[:2], np.uint8)
img_left = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)

thresh, value = Trackbar(img, img_left)
print(value)

