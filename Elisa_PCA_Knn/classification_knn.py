# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:54:38 2021

@author: Elisa
"""

import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import scipy.linalg as linalg
import numpy as np

import cv2
from scipy.linalg import svd

X_train = []
y_train = []
X_test = []
y_test = []

#Begin with mug 

N_mug = 43;
file_name = []

for i in range (N_mug) : 
    file_name.append("dataset/mug/mug"+str(i)+".jpg")

for i in range(N_mug) :
    img = cv2.imread(file_name[i])
    #plt.imshow(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(256,256))
    h,w = gray.shape
    # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
    X_train.append(gray.reshape(h*w))
    y_train.append(0) #0 = the object is a mug



#Then books
    
N_book = 22
file_name = []
for i in range(N_book) :
    file_name.append("dataset/book/book"+str(i)+".jpg")

for i in range(N_book) :
    img = cv2.imread(file_name[i])
    #plt.imshow(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(256,256))
    h,w = gray.shape
    # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
    X_train.append(gray.reshape(h*w))
    y_train.append(1) #0 = the object is a mug
    
    
X_train = np.array(X_train)
y_train = np.array(y_train)

N,M = X_train.shape

#Scaling the data

m = X_train.mean(0)
m1 = np.ones((N, 1))*m
X_train1 = X_train - m1

X_train1 = X_train1*(1/np.std(X_train1,0))

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


#This shows that we need about K = 25 component to have 90% of the variance


#Let's apply KNN with K = 28 

k=150
V = V.T
# Project data onto principal component space,
X_train2 = X_train1 @ V[:,:k]

knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train2,y_train.ravel())

#Need a test now

N_test = 10
file_name = []
for i in range(N_test) :
    file_name.append("dataset/test/test"+str(i)+".PNG")

for i in range(N_test) :
    img = cv2.imread(file_name[i])
    #plt.imshow(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(256,256))
    h,w = gray.shape
    # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
    X_test.append(gray.reshape(h*w)) 
    
X_test = np.array(X_test)
y_test = np.array([1,1,1,1,1,1,0,0,0,0])
X_test1 = X_test - np.ones((N_test, 1))*m
X_test1 = X_test1*(1/np.std(X_train1,0))
X_test2 = X_test1 @ V[:,:k]

y_estimated = knn_classifier.predict(X_test2)
y_estimated = y_estimated.T