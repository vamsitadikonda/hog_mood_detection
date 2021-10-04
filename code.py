from __future__ import division
import cv2
import os
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn import metrics


def facecrop(image):
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
    img = cv2.imread(image)
    minisize = (img.shape[1], img.shape[0])
    miniframe = cv2.resize(img, minisize)
    faces = cascade.detectMultiScale(miniframe)
    for f in faces:
        x, y, w, h = [v for v in f]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255))
        sub_face = img[y:y + h, x:x + w]
        # face_file_name = "crop.jpg"
        # cv2.imwrite(face_file_name, sub_face)
        # cv2.imshow(image, sub_face)
        # cv2.imshow(image, img)
    return sub_face


def extracthog(image):
    croppedface = facecrop(image)
    winStride = (8, 8)
    padding = (8, 8)
    locations = ((10, 20),)
    hog = cv2.HOGDescriptor()
    hist = hog.compute(croppedface, winStride, padding, locations)
    hist = hist.transpose()
    hist = hist[0]
    return hist

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print("Accuracy on training set:")
    print(clf.score(X_train, y_train))
    print("Accuracy on testing set:")
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)

    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))


imageformat = ".tiff"
basepath = "imgdata/traindata/"
traindata = []
labels = []
i = 1
while (i < 8):
    path = basepath + str(i) + "/"
    imfilelist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(imageformat)]
    count = 0
    for el in imfilelist:
        traindata.append(extracthog(el))
        labels.append(i)
        count += 1
        print(path + str(count))
    i += 1

X_train = np.asarray(traindata)
y_train = np.asarray(labels)

testpath = "imgdata/testdata/"
testdata = []
testlabels = []
i = 1
while (i < 8):
    path = testpath + str(i) + "/"
    imfilelist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(imageformat)]
    count = 0
    for el in imfilelist:
        testdata.append(extracthog(el))
        testlabels.append(i)
        count += 1
        print(path + str(count))
    i += 1

X_test = np.asarray(testdata)
y_test = np.asarray(testlabels)

clf1 = svm.SVC(kernel='linear', C=1)
train_and_evaluate(clf1, X_train, X_test, y_train, y_test)




