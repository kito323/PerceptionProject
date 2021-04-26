import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt
from sklearn.decomposition import PCA
from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml

import pandas as pd
import cv2
import glob
import os
os.environ['DISPLAY'] = ':0'


def show_some_digits(images, targets, sample_size=24, title_text='{}' ):
    '''
    Visualize random digits in a grid plot
    images - array of flatten gidigs [:,784]
    targets - final labels
    '''
    nsamples=sample_size
    rand_idx = np.random.choice(images.shape[0], nsamples)
    print(rand_idx)
    images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))


    img = plt.figure(1, figsize=(15, 12), dpi=160)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(nsamples/5.0), 5, index + 1)
        plt.axis('off')
        #each image is flat, we have to reshape to 2D array 28x28-784
        plt.imshow(image.reshape(256,256,3), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(title_text.format(label))
    plt.show()


""" main """

images = glob.glob('dataset/*.jpg')
assert images

# Read Images
ImageDataset =[]
for fname in sorted(images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(256,256))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h,w,d = img.shape
    # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
    ImageDataset.append(img.reshape(h*w*d))

ImageDataset = np.asarray(ImageDataset)

# # # Save the dataset
np.savez("ImageDataset", ImageDataset)
# # # Load the dataset
data = np.load("ImageDataset.npz")
ImageDataset = data['arr_0']

# PCA is affected by scale so you need to scale the features in the data before applying PCA
from sklearn.preprocessing import StandardScaler
ImageDataset_Scaled = StandardScaler().fit_transform(ImageDataset)
# split data 
from sklearn.model_selection import train_test_split

# Import label
lable = np.loadtxt('./lable.txt', dtype = 'str')
y = lable

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(ImageDataset_Scaled, y, test_size=0.2)

pca = PCA(0.95)  # It means that scikit-learn choose the minimum number of principal components 

principalComponents = pca.fit_transform(ImageDataset_Scaled)

print(pca.explained_variance_ratio_)
print(len(pca.explained_variance_ratio_))
print(sum(pca.explained_variance_ratio_))

## Apply Logistic Regression to the Transformed Data
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter = 10000)
logisticRegr.fit(X_train, y_train)


# Predict for One Observation (image)
print(logisticRegr.predict(X_test[0:10]))
plt.imshow(X_test[3].reshape(256,256,3).astype('uint8'), interpolation='nearest')


## TEST 
print(logisticRegr.predict(X_test))
# show_some_digits(X_test, logisticRegr.predict(X_test), 14)
# show_picture(X_test, logisticRegr.predict(X_test), 13)
plt.imshow(X_test[12].reshape(256,256,3), cmap=plt.cm.gray_r, interpolation='nearest')