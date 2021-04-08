# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:24:19 2021

@author: kuion
"""

import numpy as np
import cv2


def FindAllChessCorners(nrX, nrY, imgLocLst):
    """
    nrX; nrY - number of chessboard corners
    imgLocLst - list of locations of images
    """
    
    nb_horizontal = nrY
    nb_vertical = nrX
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
    objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = imgLocLst
    
    for fname in images:
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        ret, corners = cv2.findChessboardCorners(img, (nb_vertical,nb_horizontal),  None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            imgpoints.append(corners)
    
            # # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)
    #cv2.destroyAllWindows()
    
    return (objpoints, imgpoints)



