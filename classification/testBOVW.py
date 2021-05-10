import cv2
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn import svm, datasets

from classification.utilsBOVW import *


def predictLabel(img, sift, num_clusters, kmeans, svm, scaler, imgs_features):
    name_dict =	{
        "0": "book",
        "1": "box",
        "2": "mug",
    }
    # Convert to gray and resize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[0] > 256 and img.shape[1] > 256:
        img = cv2.resize(img, (256,256))

    # Get descriptors
    kp, des = sift.detectAndCompute(img, None)
    # Stack descriptors vertically
    descriptors = vstackDescriptors(des)
    # Extract features
    test_features = extractFeatures(kmeans, [des], 1, num_clusters)
    # Scale feature
    test_features = scaler.transform(test_features)
    kernel_test = np.dot(test_features, imgs_features.T)
    # Make prediction
    prediction = [name_dict[str(int(i))] for i in svm.predict(kernel_test)]
    return prediction
