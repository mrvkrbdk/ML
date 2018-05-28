import numpy
import cv2
import time
import csv
import matplotlib.pyplot as plt
from  sklearn import linear_model as lr
import pandas as pd
import  os

# resimleri64*64yaptım,16*16lık bloklarla(yani8*8likhucreler)yani yatayda ve dikeyde 8er tane blok var yataydaki her blokta 7kezgezerdikeydede7keztoplmda49gezme her bolkta 8*8lik 4hucre var sonucta 9histogram*4lükhücreden sonucta 36 deger var 49gezme*36 degerden 1764 veri elde ederiz
face_cascade = cv2.CascadeClassifier('/home/mervek/Desktop/haarcascade_frontalface_default.xml')
winSize = (64, 64)
blockSize = (16, 16) # h x w in cells
blockStride = (8, 8)
cellSize = (8, 8)   # h x w in pixels
nbins = 9  # number of orientation bins
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

def matrixdondur(path):           #video yolu verip yakalanmış yüzlerden her resimin hoglu hlini matrix ile döndürüyoruz

    vidcap = cv2.VideoCapture("/home/mervek/Desktop/bitirme/video_/%s" %path)
    success, img = vidcap.read()
    # k_image=cv2.resize(image,(224,224))
    hist2=[]

    count = 1
    success = True
    while success:
        faces = face_cascade.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
        #print('facessss', faces)


        for (x, y, w, h) in faces:
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv2.imwrite("/home/mervek/Desktop/bitirme/frame_/withoutcrop/%d.jpg" % (count), image)

            crop_img = image[y:y + h, x:x + w]
            k_image = cv2.resize(crop_img, (64, 64))
            cv2.imwrite("/home/mervek/Desktop/bitirme/frame_/videokirp/%d.jpg" %(count), k_image)
            hist=hog.compute(k_image)
            hist2.append(hist)

        success, img = vidcap.read()
        #print('Read a new frame: ', success)
        #uzunluk=len(hist2[count-1])
        #print('frame uzunluk: ', uzunluk)

        #print(hist2[count-1])
        count += 1
    mat=numpy.asarray(hist2)
    return mat



def funcs(mat):
    lenght = len(mat[0])
    l = mat[:, 0]
    a = len(l)
    c = 1
    n = []
    while c != a + 1:
        n.append(c)
        c += 1
    x = n[:]

    pol = numpy.zeros(lenght)
    c2 = 0
    while c2 != len(mat[0]):
        y = mat[:, c2]
        z = numpy.polyfit(x, y, 2)
        m = z[0]
        pol[c2] = m
        c2 += 1
    print(z)
    ext = numpy.zeros(5 * lenght)
    ext[0:lenght] = numpy.squeeze(mat.mean(0))
    ext[lenght:lenght * 2] = numpy.squeeze(mat.std(0))
    ext[lenght * 2:lenght * 3] = numpy.squeeze(mat.min(0))
    ext[lenght * 3:lenght * 4] = numpy.squeeze(mat.max(0))
    ext[lenght * 4:lenght * 5] = pol

    return ext
def videoal(name):
    mins = 0
    cap=cv2.VideoCapture(0)
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter('/home/mervek/Desktop/bitirme/video_/%s'%name,fourcc,20.0,(640,480))
    while mins != 100:
        ret,frame=cap.read()
        out.write(frame)
        cv2.imshow('frame',frame)
        mins += 1

        k=cv2.waitKey(1) & 0xFF
        if k == 5:
            break

    cap.release()  #webcam calıştı
    out.release()  #webcam cıktısını al
    cv2.destroyAllWindows()
    return name

#


import pandas as pd
import  numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ext=funcs(matrixdondur(videoal("output9.avi")))
#ext=funcs(matrixdondur(""))

dataset=pd.read_csv("/home/mervek/Desktop/bitirme/csv/hogmmiwithpolyfitparameter2.csv",header=None)
array = dataset.values
X = array[:,0:8820]
Y = array[:,8820]
print(dataset.head())
print(Y)
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
# Spot Check Algorithms


lr=LogisticRegression(C=6) #42,44,44,45,45,45
#lr=SVC(C=6.0,kernel='linear')

#models.append(('SVM', SVC(C=6.0,kernel='linear'))) #43,43,43,43,43,43

# evaluate each model in turn

kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(lr, X_train, Y_train, cv=kfold, scoring=scoring)
msg = "logistic R: ortalama= %f standart sapma =(%f)\n" % (cv_results.mean(), cv_results.std())
print(msg)

lr.fit(X_train,Y_train)
predictions=lr.predict(X_validation)
print("dogruluk oranı: ",accuracy_score(Y_validation, predictions))
print("karşılaştırma matrisi :\n ",confusion_matrix(Y_validation, predictions))
print("sınıflandırma sonuçları:\n",classification_report(Y_validation, predictions))



prediction_video=lr.predict([ext])
print("video tahmin etiketi :",prediction_video)

