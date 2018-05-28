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

def withoutcrope(path):           #video ismini path e verip her frame için histogram oluşturup csv ye yazıyoruz

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
            k_image = cv2.resize(crop_img, (224, 224))
            cv2.imwrite("/home/mervek/Desktop/bitirme/frame_/videokirp/%d.jpg" %(count), k_image)
            hist=hog.compute(k_image)
            hist2.append(hist)

        success, img = vidcap.read()
        print('Read a new frame: ', success)
        print('frame uzunluk: ', len(hist2[count-1]))
        print(hist2[count-1])
        count += 1
    with open("/home/mervek/Desktop/bitirme/csv/%s.csv" % path, "a") as path:
        wr = csv.writer(path)
        c = 0
        while c != count - 1:
            wr.writerow(hist2[c])
            c+=1

    return count


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


def hogla(path,framesayısı):   #keilmişkırpışmış framelerin oldugu dosya yolunu ve içersindeki frame sayısını vererk her resimin hoglu hlini matrix ile döndürüyoruz

    n=1    #mmı 1den baslatmıs bizimkiler 0
    mat = []
    while n!=framesayısı:
        img = cv2.imread("%s/%d.jpg" % (path, n))
        r_img=cv2.resize(img, (64, 64))
        mat.append(hog.compute(r_img))
        x = numpy.asarray(mat)
        n+=1

    #print(len(x[0]))
    return x


def ful(path):           #video yolu verip yakalanmış yüzlerden her resimin hoglu hlini matrix ile döndürüyoruz

    vidcap = cv2.VideoCapture("/home/mervek/Desktop/bitirme/video_/%s" %path)
    success, img = vidcap.read()
    # k_image=cv2.resize(image,(224,224))
    hist2=[]
    dizi2=[]
    count = 1
    success = True
    while success:
        faces = face_cascade.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
        #print('facessss', faces)


        for (x, y, w, h) in faces:
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv2.imwrite("/home/mervek/Desktop/bitirme/frame_/withoutcrop/%d.jpg" % (count), image)

            crop_img = image[y:y + h, x:x + w]
            k_image = cv2.resize(crop_img, (224, 224))
            cv2.imwrite("/home/mervek/Desktop/bitirme/frame_/videokirp/%d.jpg" %(count), k_image)
            hist=hog.compute(k_image)
            hist2.append(hist)

        success, img = vidcap.read()
        print('Read a new frame: ', success)
        uzunluk=len(hist2[count-1])
        print('frame uzunluk: ', uzunluk)
        dizi2.append(count)
        count += 1
    dizi2.append(count)
    print(dizi2)

    mat=numpy.asarray(hist2)
    x = dizi2[:]
    y = mat[:, 0]
    z = numpy.polyfit(x, y, 2)
    p = numpy.poly1d(z)
    xp = numpy.linspace(1, 150,100)
    print(z)
    plt.plot(x,y,'.', xp, z, '-')
    plt.show()


#ful("videoplayback.mp4")


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
        z = numpy.polyfit(x, y, 3)
        m = z[0]
        pol[c2] = m
        c2 += 1

    ext = numpy.zeros(5 * lenght)
    ext[0:lenght] = numpy.squeeze(mat.mean(0))
    ext[lenght:lenght * 2] = numpy.squeeze(mat.std(0))
    ext[lenght * 2:lenght * 3] = numpy.squeeze(mat.min(0))
    ext[lenght * 3:lenght * 4] = numpy.squeeze(mat.max(0))
    ext[lenght * 4:lenght * 5] = pol

    print(ext)

#funcs(matrixdondur("emotionvideo.avi"))      #"""""""""""""""""""""""""""""""""""""""""""""""""""""""predicte burayı ver""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def f(mat,s):           #dosya içindeki videodan döndürdüğümüz bütün resimlerin tek tek hoglanmış özniteliklerden özetleme (ort,std,max,min,polyfit egimi) s parametresi mmi etiketi için
    lenght=len(mat[0])
    l=mat[:,0]
    a=len(l)
    c = 1
    n = []
    while c != a+1:
        n.append(c)
        c += 1
    x = n[:]

    pol = numpy.zeros(lenght)
    c2=0
    while c2!=len(mat[0]):
        y=mat[:,c2]
        z = numpy.polyfit(x, y, 4)
        m=z[0]
        pol[c2]=m
        c2+=1

    ext=numpy.zeros(5*lenght+1)
    ext[0:lenght]=numpy.squeeze(mat.mean(0))
    ext[lenght:lenght*2] = numpy.squeeze(mat.std(0))
    ext[lenght*2:lenght*3] = numpy.squeeze(mat.min(0))
    ext[lenght*3:lenght*4] = numpy.squeeze(mat.max(0))
    ext[lenght*4:lenght*5] = pol
    ext[lenght * 5:lenght * 5+1]=s                            #etiketleri ekledin unutma!!!  sadece model egitimi için oluşturduk yani etiketlerle beraber 8820+1 nitelikten oluşşan 1 öznitelik çıkarttık

    print(len(ext))
    with open("/home/mervek/Desktop/bitirme/csv/foronepersonhogmmiwithpolyfitparameter4.csv", "a") as fp:
         wr = csv.writer(fp)
         wr.writerow(ext)

#f(matrixdondur("videoplayback.mp4"))




def dataext(path):
    #etiketleri ekliyorum
    dataset = pd.read_csv("/home/mervek/Desktop/csvforsubj.csv")
    array = dataset.values
    X = array[:, 0:]
    ee = numpy.zeros(12) #207
    ee[0:len(ee)] = numpy.squeeze(X)
    #

    li=os.listdir(path)
    ar=numpy.asarray(li)
    x=sorted(ar)
    c=len(x)
    n=0
    while n!=c:
        f(hogla(path+x[n],len(os.listdir(path+x[n]))),ee[n])
        n+=1

#dataext("/home/mervek/Desktop/CKP/Untitled Folder/MMI/MMI_DPM_alignment/")


dataext("/home/mervek/Desktop/oneperson/")



