import cv2
import numpy as np 
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump, load

from utilsBOVW import *

def main ():

    train_path = 'dataset/train'

    # read images for training
    images = getImages(train_path)
    # Create sift object
    sift = cv2.xfeatures2d.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])
    label_count = 3
    image_count = len(images)
    
    # read all the images and classify them (add a label)
    for img_path in images:
        if("book" in img_path):
            class_index = 0
        elif("box" in img_path):
            class_index = 1
        elif("mug" in img_path):
            class_index = 2

        train_labels = np.append(train_labels, class_index)
        img = resizeImg(img_path)
        # get descriptor and keypoint of the img
        kp, des = sift.detectAndCompute(img, None)
        descriptor_list.append(des)

    # Stack descriptors vertically in a numpy array 
    descriptors = vstackDescriptors(descriptor_list)

    # Apply kmeans to the descriptors
    num_clusters = 100 # num of centroids
    n_init = 3 # number of times kmeans will run with different centroids
    kmeans = KMeans(n_clusters = num_clusters, n_init = n_init).fit(descriptors)

    # Extract features
    imgs_features = extractFeatures(kmeans, descriptor_list, image_count, num_clusters)

    # Scale features
    scaler = StandardScaler().fit(imgs_features)        
    imgs_features = scaler.transform(imgs_features)

    # Apply SVM to the features
    svm = applySVM(imgs_features, train_labels)

    # Save the trained model
    dump((svm, kmeans, scaler, no_clusters, imgs_features), "BOVW.pkl", compress=3) 

    # Plot Features Histogram
    plotHistogram(imgs_features, num_clusters)


if __name__ == '__main__':
    main()