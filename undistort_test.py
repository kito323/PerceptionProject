# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 23:25:49 2021

@author: krist
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

nb_horizontal = 6
nb_vertical = 9


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# Arrays to store object points and image points from all the images. 
objpoints2 = [] 
imgpoints2 = [] 


images_right = glob.glob('data/imgs/calibration/Stereo_calibration_images/right-*.png')


assert images_right
for fname in images_right:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints2.append(objp)
        imgpoints2.append(corners)

        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()


ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints2, imgpoints2, gray.shape[::-1], None, None)

#%%

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx_right,dist_right,gray.shape[::-1],1,gray.shape[::-1]) #This keeps the black parts

img_right = cv2.imread('data/imgs/calibration/Stereo_calibration_images/right-0045.png')
dst_right = cv2.undistort(img_right, mtx_right, dist_right, None, newcameramtx)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,18))
ax[0].imshow(img_right[...,[2,1,0]])
ax[0].set_title('Original image right')
ax[1].imshow(dst_right[...,[2,1,0]])
ax[1].set_title('Undistorted image right')

#%%
#plt.imshow(dst_right[600:700,1100:1300])