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

def dispTrackbar(gray1, gray2, BMvSGBM, def_val=[0,5,6,10,100,32,150,1]):
    win_name = "ThresholdTrackbar"
    img_win_name = "Result"

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(img_win_name, cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow(win_name, 900,400) #1000, 500
    cv2.resizeWindow(img_win_name, 700,600)
    
    cv2.createTrackbar("min_disp", win_name, def_val[0], 255, empty_function)
    cv2.createTrackbar("block_size", win_name, def_val[1], 255, empty_function)
    cv2.createTrackbar("num_disp", win_name, def_val[2], 255, empty_function)
    cv2.createTrackbar("uniqueness", win_name, def_val[3], 255, empty_function)
    cv2.createTrackbar("texture_thresh", win_name, def_val[4], 255, empty_function)
    cv2.createTrackbar("speckle_range", win_name, def_val[5], 255, empty_function)
    cv2.createTrackbar("speckle_window", win_name, def_val[6], 255, empty_function)
    cv2.createTrackbar("disp12maxdiff", win_name, def_val[7], 255, empty_function)
    
    while True:
        min_disp = cv2.getTrackbarPos("min_disp", win_name)
        block = cv2.getTrackbarPos("block_size", win_name)
        num_disp = cv2.getTrackbarPos("num_disp", win_name)
        uniq = cv2.getTrackbarPos("uniqueness", win_name)
        texture = cv2.getTrackbarPos("texture_thresh", win_name)
        specle_range = cv2.getTrackbarPos("speckle_range", win_name)
        specle_window = cv2.getTrackbarPos("speckle_window", win_name)
        disp12maxdiff = cv2.getTrackbarPos("disp12maxdiff", win_name)
        
        if BMvSGBM == "SGBM":
            win_size = 5
            stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
                                           numDisparities = 16*(num_disp+1),
                                           blockSize = 5+2*block,
                                           uniquenessRatio = uniq,
                                           speckleWindowSize = specle_window,
                                           speckleRange = specle_range,
                                           disp12MaxDiff = disp12maxdiff,
                                           P1 = 8*3*win_size**2,
                                           P2 =32*3*win_size**2)
        else:
            stereo = cv2.StereoBM_create(numDisparities=16*(num_disp+1), blockSize=5+2*block)
            stereo.setMinDisparity(min_disp)
            stereo.setUniquenessRatio(uniq)
            stereo.setTextureThreshold(texture)
            stereo.setSpeckleRange(specle_range)
            stereo.setSpeckleWindowSize(specle_window)
            stereo.setDisp12MaxDiff(disp12maxdiff)
        
        
        disparity = stereo.compute(gray1, gray2)
        disparity2 = cv2.normalize(disparity, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow(img_win_name , disparity2)

        # Code exits "while true loop" by pressing letter 'c'
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    values = [min_disp, block, num_disp, uniq, texture, specle_range, specle_window]
    return disparity, values

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

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

#read in 
img_left = cv2.imread(images_left[1400])
img_right = cv2.imread(images_right[1400])
    
#rectify
imgU1 = np.zeros(img_left.shape[:2], np.uint8)
imgU1 = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
imgU2 = np.zeros(img_right.shape[:2], np.uint8)
imgU2 = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR, imgU2, cv2.BORDER_CONSTANT, 0)

#make gray
gray1 = cv2.cvtColor(imgU1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(imgU2, cv2.COLOR_BGR2GRAY)


disparity, values = dispTrackbar(gray1, gray2, "BM" ,[0, 1, 12, 4, 253, 157, 147, 1])
print(values)
#disparity, values = dispTrackbar(imgU1, imgU2, "SGBM" ,[0, 1, 12, 4, 253, 157, 147, 1])
#print(values)

#%%

points_3D = cv2.reprojectImageTo3D(disparity, Q)

colors = cv2.cvtColor(imgU1, cv2.COLOR_BGR2RGB)
mask = (disparity > 0)#disparity.min()) & (disparity != 0)
out_points = points_3D[mask]
out_colors = colors[mask]
out_fn = 'out.ply'
write_ply(out_fn, out_points, out_colors)


source = o3d.io.read_point_cloud("out.ply")
source.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([source])

#disparity = stereo.compute(gray1, gray2)
#plt.imshow(cv2.cvtColor(imgU1, cv2.COLOR_BGR2RGB))
#plt.imshow(disparity, cmap='gray')
#plt.show()

#%% Havent gotten past this point yet because the disparity looks bad
for i in range(len(images_left)):
    
    #read in 
    img_left = cv2.imread(images_left[i])
    img_right = cv2.imread(images_right[i])
    
    #rectify
    imgU1 = np.zeros(img_left.shape[:2], np.uint8)
    imgU1 = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
    imgU2 = np.zeros(img_right.shape[:2], np.uint8)
    imgU2 = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR, imgU2, cv2.BORDER_CONSTANT, 0)
    
    
    
    fgmask = fgbg.apply(imgU1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #Draw only the biggest contour if its size is over threshold
    if len(cnts) != 0:
        c = max(cnts, key = cv2.contourArea)
        if cv2.contourArea(c) > 4000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(imgU1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Video", imgU1)
    cv2.imshow("Thresh", fgmask)
    
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()