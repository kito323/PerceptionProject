# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 08:38:48 2021

@author: Elisa
"""
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import scipy.linalg as linalg
import numpy as np

import cv2
from scipy.linalg import svd

#take only the edge

book = cv2.imread("dataset2/book/book (1).jpg")

book = cv2.cvtColor(book,cv2.COLOR_BGR2GRAY)
book = cv2.resize(book,(256,256))
plt.figure()
plt.imshow(book,cmap='gray')

threshold =50
threshold_value =255
thresh = cv2.adaptiveThreshold(book, threshold, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
plt.imshow(thresh,cmap='gray')



####Test on the data

X_train = []
y_train = []
X_test = []
y_test = []

N_mug = 87;
file_name = []

for i in range (N_mug) : 
    file_name.append("dataset2/mug2/mug ("+str(i+1)+").jpg")

for i in range(N_mug) :
    img = cv2.imread(file_name[i])
    #plt.imshow(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(256,256))
    thresh = cv2.adaptiveThreshold(gray, threshold, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

    
    h,w = thresh.shape
    # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
    X_train.append(thresh.reshape(h*w))
    y_train.append(0) #0 = the object is a mug



#Then books
    
N_book = 124
file_name = []
for i in range(N_book) :
    file_name.append("dataset2/book/book ("+str(i+1)+").jpg")

for i in range(N_book) :
    img = cv2.imread(file_name[i])
    #plt.imshow(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(256,256))
    thresh = cv2.adaptiveThreshold(gray, threshold, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

    thresh = cv2.resize(thresh,(256,256))
    h,w = thresh.shape
    
    # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
    X_train.append(thresh.reshape(h*w))
    y_train.append(1)
    

X_train = np.array(X_train)
y_train = np.array(y_train)

N,M = X_train.shape

#Scaling the data

m = X_train.mean(0)
m1 = np.ones((N, 1))*m
X_train1 = X_train - m1
#X_train1 = X_train1*(1/np.std(X_train1,0))

# PCA by computing SVD of X1
U,S,V = svd(X_train1,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.90

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

k=200
V = V.T
# Project data onto principal component space,
X_train2 = X_train1 @ V[:,:k]

knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train2,y_train.ravel())

n=[0,1]
# Plot PCA of the data
f = plt.figure()
plt.title(' ')
for c in n:
    # select indices belonging to class c:
    class_mask = (y_train == c)
    plt.plot(X_train2[class_mask,0], X_train2[class_mask,1], 'o')
plt.legend(['0','1'])
plt.xlabel('PC1')
plt.ylabel('PC2')

K_vis=16
# Visualize the pricipal components
plt.figure(figsize=(8,6))
for k in range(K_vis):
    N1 = np.ceil(np.sqrt(K_vis)); N2 = np.ceil(K_vis/N1)
    plt.subplot(N2, N1, k+1)
    I = np.reshape(V[:,k], (256,256))
    plt.imshow(I, cmap=plt.cm.hot)
    plt.title('PC{0}'.format(k+1));
    
N_test = 28
file_name = []
for i in range(N_test) :
    file_name.append("dataset2/test3/test"+str(i)+".jpg")

for i in range(N_test) :
    img = cv2.imread(file_name[i])
    #plt.imshow(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(256,256))
    thresh = cv2.adaptiveThreshold(gray, threshold, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)

    thresh = cv2.resize(thresh,(256,256))
    h,w = thresh.shape
    
    # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
    X_test.append(thresh.reshape(h*w)) 
    
X_test = np.array(X_test)
y_test = np.array([0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0])
X_test1 = X_test - np.ones((N_test, 1))*X_test.mean(0)

k=200
X_test2 = X_test1 @ V[:,:k]

y_estimated = knn_classifier.predict(X_test2)
y_estimated = y_estimated.T

error = sum(abs(y_estimated-y_test))

# Plot PCA of the data
f = plt.figure()
plt.title(' ')
for c in n:
    # select indices belonging to class c:
    class_mask = (y_test == c)
    plt.plot(X_test2[class_mask,0], X_test2[class_mask,1], 'o')
plt.legend(['0','1'])
plt.xlabel('PC1')
plt.ylabel('PC2')