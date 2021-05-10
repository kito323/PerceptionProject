import glob
import imutils
import os
import cv2
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from joblib import load

from classification.utilsBOVW import *
from classification.testBOVW import predictLabel

os.environ['DISPLAY'] = ':0'

def main():

    images_left = glob.glob('data/imgs//withoutOcclusions/left/*.png')
    # images_left = glob.glob('data/imgs//withOcclusions/left/*.png')

    map1x = np.loadtxt('data/map1x.csv', delimiter = "\t").astype("float32")
    map1y = np.loadtxt('data/map1y.csv', delimiter = "\t").astype("float32")

    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False, history = 600, varThreshold = 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

    # LOAD CLASSIFICATOR
    svm, kmeans, scaler, num_cluster, imgs_features = load('classification/BOVW_100_mix.pkl')
    # Create sift object
    sift = cv2.xfeatures2d.SIFT_create()

    # define conveyor_area
    conveyor_area = np.array([ [[395,520]], [[447,687]], [[1279,373]], [[1127, 312]] ])
    # define conveyor_area
    conveyor_area = np.array([ [[393,515]], [[443,691]], [[1279,369]], [[1130, 306]] ])


    assert images_left
    i = 0
    arr_label = []
    found = False
    for fname in sorted(images_left):
        #read in 
        img = cv2.imread(fname)
        
        #rectify
        imgU1 = np.zeros(img.shape[:2], np.uint8)
        imgU1 = cv2.remap(img, map1x, map1y, cv2.INTER_LINEAR, imgU1, cv2.BORDER_CONSTANT, 0)
        
        fgmask = fgbg.apply(imgU1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        #Draw only the biggest contour if its size is over threshold
        if len(cnts) != 0:
            c = max(cnts, key = cv2.contourArea)
            if cv2.contourArea(c) > 3500:# and cv2.contourArea(c) <00:
                (x, y, w, h) = cv2.boundingRect(c)
                center = calculate_rect_center(x, y, w, h)
                print(center)

                if cv2.pointPolygonTest(conveyor_area, center, measureDist = False) == 1:
                    # Image to predict
                    img = imgU1[y : y+h, x : x+w]
                    print(img.shape)
                    label = predictLabel(img, sift, num_cluster, kmeans, svm, scaler, imgs_features)
                    cv2.rectangle(imgU1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if i < 11:
                        arr_label.append(label)
                        i += 1
                    elif i == 11:
                        text = max_occurences(arr_label) 
                        cv2.putText(imgU1, text, (x, y - 20), cv2.FONT_ITALIC, 0.75, (0,0,255), 1, cv2.LINE_AA)
                        i = 0
                        arr_label.clear()
                        found = True
                    if found:
                        cv2.putText(imgU1, text, (x, y - 20), cv2.FONT_ITALIC, 0.75, (0,0,255), 1, cv2.LINE_AA)


        cv2.imshow("Video", imgU1)
        cv2.imshow("Thresh", fgmask)
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
    cv2.destroyAllWindows()

# Find the max in the number of predictions
def max_occurences(array):
    count_book = array.count(['book'])
    count_box = array.count(['box'])
    count_mug = array.count(['mug'])
    value = [count_book, count_box, count_mug]
    if value.index(max(value)) == 0:
        return 'Book'
    elif value.index(max(value)) == 1:
        return 'Box'
    else:
        return 'Mug'


# find if the rectangular contour is inside the area depicted by the conveyor
def calculate_rect_center(x,y,w,h):
    c_x = int(x + w/2)
    c_y = int(y + h/2)
    return (c_x,c_y)


if __name__ == '__main__':
    main()