#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# # Weekly project part 1
#     Using the image "appletree.jpg"
#     A) Can you segment the apples from the tree?
#     B) Can you get the computer to count how many there are? 
#         How close can you get there are 26.
#     C) Can you change color of one of them?
#     D) Can you segment the leaves?
#     
#     

# ## A)

# In[2]:


path = "appletree.jpg"
bgr_img = cv2.imread(path)

plt.imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))


# In[3]:


plt.imshow(bgr_img[:,:,-1], cmap="gray")


# Thresholding a single channel won't probably do the job...

# In[8]:


def empty_function(*args):
    pass

def ThresholdTrackbar(img):
    win_name = "ThresholdTrackbar"
    img = img.copy()

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 20,4)
    
    cv2.createTrackbar("threshold_B", win_name, 0, 255, empty_function)
    cv2.createTrackbar("threshold_G", win_name, 0, 255, empty_function)
    cv2.createTrackbar("threshold_R", win_name, 0, 255, empty_function)

    while True:
        th_B = cv2.getTrackbarPos("threshold_B", win_name)
        th_G = cv2.getTrackbarPos("threshold_G", win_name)
        th_R = cv2.getTrackbarPos("threshold_R", win_name)
        
        img_temp = img.copy()
        (b, g, r) = cv2.split(img_temp)
        thresholded_B = cv2.threshold(b, th_B, 255, cv2.THRESH_BINARY)[1]
        thresholded_G = cv2.threshold(g, th_G, 255, cv2.THRESH_BINARY)[1]
        thresholded_R = cv2.threshold(r, th_R, 255, cv2.THRESH_BINARY)[1]
        bgr_thresh = cv2.merge((thresholded_B, thresholded_G, thresholded_R))
        cv2.imshow(win_name, bgr_thresh)
        
        # Code exits "while true loop" by pressing letter 'c'
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            break

    cv2.destroyAllWindows()
    return bgr_thresh


# In[9]:


th = ThresholdTrackbar(bgr_img)
plt.imshow(cv2.cvtColor(th, cv2.COLOR_BGR2RGB))


# # Weekly project part 2
#     A) Remove the greenscreen and replace the background in "itssp.png"?
#     B) Can improve the edge with erroding/dialating?

# In[ ]:




