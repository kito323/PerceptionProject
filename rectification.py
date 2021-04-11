# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 16:23:45 2021

@author: krist
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

"""
Implement the number of vertical and horizontal corners
"""
nb_horizontal = 6
nb_vertical = 9


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints1 = [] 
objpoints2 = [] 
imgpoints1 = [] 
imgpoints2 = [] 


images_left = glob.glob('data/imgs/calibration/Stereo_calibration_images/left-*.png')
images_right = glob.glob('data/imgs/calibration/Stereo_calibration_images/right-*.png')

assert images_left
for fname in images_left:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints1.append(objp)
        imgpoints1.append(corners)

        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()


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

ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints1, imgpoints1, gray.shape[::-1], None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints2, imgpoints2, gray.shape[::-1], None, None)

retval, mtx_left, distCoeffs1, mtx_right, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints2, imgpoints1, imgpoints2, mtx_left, dist_left, mtx_right, dist_right, gray.shape[::-1])

#%%

print(mtx_left)
print(mtx_right)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_left, distCoeffs1, mtx_right, distCoeffs2, gray.shape[::-1], R, T, None, None, None, None)

map1x, map1y = cv2.initUndistortRectifyMap(mtx_left, distCoeffs1, R1, P1, gray.shape[::-1], cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx_right, distCoeffs2, R2, P2, gray.shape[::-1], cv2.CV_32FC1)


img_left = cv2.imread('data/imgs/calibration/Stereo_calibration_images/left-0030.png')
img_right = cv2.imread('data/imgs/calibration/Stereo_calibration_images/right-0030.png')

imgU1 = np.zeros(gray.shape, np.uint8)
imgU1 = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)

imgU2 = np.zeros(gray.shape, np.uint8)
imgU2 = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR, imgU2, cv2.BORDER_CONSTANT, 0)

print(map1x)
print(distCoeffs2)

#dst_left = cv2.undistort(img_left, mtx_left, dist_left, None, None)
#dst_right = cv2.undistort(img_right, mtx_right, dist_right, None, None)

#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,18))
#ax[0].imshow(img_left[...,[2,1,0]])
#ax[0].set_title('Original image left')
#ax[1].imshow(dst_left[...,[2,1,0]])
#ax[1].set_title('Undistorted image left')

#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,18))
#ax[0].imshow(img_right[...,[2,1,0]])
#ax[0].set_title('Original image right')
#ax[1].imshow(dst_right[...,[2,1,0]])
#ax[1].set_title('Undistorted image right')