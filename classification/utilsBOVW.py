import cv2
import numpy as np 

def getImages(train_path):
    images = []
    count = 0
    for folder in os.listdir(train_path):
        for file in  os.listdir(os.path.join(train_path, folder)):
            images.append(os.path.join(train_path, os.path.join(folder, file)))

    np.random.shuffle(images)    
    return images

def resizeImg(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img,(256,256))

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 
    return descriptors

def extractFeatures(kmeans, descriptor_list, image_count, num_clusters):
    img_features = np.array([np.zeros(num_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            img_features[i][idx] += 1
    return img_features

def applySVM(imgs_features, train_labels):
    # define dict of parameters
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3, 1e0, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5]
    gammas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.11, 0.095, 0.105]
    param_grid = {'kernel':('linear','rbf', 'sigmoid','precomputed','poly'), 'C': Cs, 'gamma' : gammas}
    # find the best ones
    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds)
    grid_search.fit(imgs_features, train_labels)
    print(grid_search.best_params_)
    # pick the best ones
    C, gamma, kernel = grid_search.best_params_.get("C"), grid_search.best_params_.get("gamma"), grid_search.best_params_.get("kernel")  

    svm = SVC(kernel = 'kernel', C =  C_param, gamma = gamma_param, class_weight = 'balanced')
    svm.fit(imgs_features, train_labels)
    print("training score\n:", clf.score(imgs_features, train_labels))
    return svm

def plotHistogram(imgs_features, num_clusters):
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:,h], dtype=np.int32)) for h in range(no_clusters)])

    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title("Complete Vocabulary Generated")
    plt.xticks(x_scalar + 0.4, x_scalar)
    plt.show()