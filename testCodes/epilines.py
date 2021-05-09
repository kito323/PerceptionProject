# -*- coding: utf-8 -*-
"""
Created on Sun May  9 13:38:35 2021

@author: krist
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")

map2x = np.loadtxt('data/map2x.csv', delimiter = "\t").astype("float32")
map2y = np.loadtxt('data/map2y.csv', delimiter = "\t").astype("float32")

images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
images_right = glob.glob('data/imgs//withoutOcclusions/right/*.png')

img1 = cv2.imread(images_left[800], 0)
img2 = cv2.imread(images_right[800], 0)

imgU1 = np.zeros(img1.shape[:2], np.uint8)
imgU1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
img1 = imgU1

imgU2 = np.zeros(img2.shape[:2], np.uint8)
imgU2 = cv2.remap(img2, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
img2=imgU2

#%%

sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1] 


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

#%%

#colors = np.array([(255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)])

# draw the provided points on the image
def drawPoints(img, pts):
    for pt in zip(pts):
        cv2.circle(img, tuple(pt[0]), 5, 255, -1)

# draw the provided lines on the image
def drawLines(img, lines):
    _, c, _ = img.shape
    for r in zip(lines):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cv2.line(img, (x0, y0), (x1, y1), (0, 0, 0), 1)
        
# get 3 image points of interest from each image and draw them
ptsL = np.asarray([[[500, 200], [500, 400], [500, 600]]])
ptsR = np.asarray([[[700, 100], [700, 300], [700, 500]]])
drawPoints(img1, ptsL)
drawPoints(img2, ptsR)

# find epilines corresponding to points in right image and draw them on the left image
epilinesR = cv2.computeCorrespondEpilines(ptsR.reshape(-1, 1, 2), 2, F)
epilinesR = epilinesR.reshape(-1, 3)
drawLines(img1, epilinesR)

# find epilines corresponding to points in left image and draw them on the right image
epilinesL = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, F)
epilinesL = epilinesL.reshape(-1, 3)
drawLines(img2, epilinesL)

# combine the corresponding images into one and display them
#combineSideBySide(img1, img2, "epipolar_lines", save=True)

plt.subplot(121),plt.imshow(img1)
plt.subplot(122),plt.imshow(img2)
plt.show()

#fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,18))
#ax[0].imshow(img5[...,[2,1,0]])
#ax[0].set_title('Original image left')
#ax[1].imshow(img3[...,[2,1,0]])
#ax[1].set_title('After remapping left')

#%%
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.match(des1, des2)

# Sort them in the order of their distance (i.e. best matches first).
matches = sorted(matches, key = lambda x:x.distance)

nb_matches = 10

good = []
pts1 = []
pts2 = []

for m in matches[:nb_matches]:
    good.append(m)
    pts1.append(kp1[m.queryIdx].pt)
    pts2.append(kp2[m.trainIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
    
F, mask=cv2.findFundamentalMat(pts1, pts2)

# We select only inlier points
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,2)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2 ,F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)


#plt.subplot(121),plt.imshow(img5)
#plt.subplot(122),plt.imshow(img3)
plt.imshow(img5)
#plt.imshow(img3)
plt.show()


#ig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(10,10))
#axs[0, 0].imshow(img4)
#axs[0, 0].set_title('left keypoints')
#axs[0, 1].imshow(img6)
#axs[0, 1].set_title('right keypoints')
#axs[1, 0].imshow(img5)
#axs[1, 0].set_title('left epipolar lines')
#axs[1, 1].imshow(img3)
#axs[1, 1].set_title('right epipolar lines')
#plt.show()