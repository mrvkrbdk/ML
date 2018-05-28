import numpy
import cv2
import time
import csv

face_cascade = cv2.CascadeClassifier('/home/mervek/Desktop/haarcascade_frontalface_default.xml')


def videofromwebcamtoframe(min):
    mins = 0
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('/home/mervek/Desktop/bitirme/video_/outs.avi', fourcc, 20.0, (640, 480))
    while mins != min*10:
        ret, frame = cap.read()
        out.write(frame)
        cv2.imshow('frame', frame)
        mins += 1

        k = cv2.waitKey(1) & 0xFF
        if k == 5:
            break

    cap.release()  # webcam calıştı
    out.release()  # webcam cıktısını al
    cv2.destroyAllWindows()


def withoutcrope(path):           #yolu verip yakalanmış yüzlerle withoutcrape dosyasına resimleri atma  yüklediğimiz videodan kaç frame çıkarttıgını döndürüyo

    vidcap = cv2.VideoCapture("/home/mervek/Desktop/bitirme/video_/%s" %path)
    success, image = vidcap.read()
    # k_image=cv2.resize(image,(224,224))
    count = 0
    success = True
    while success:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
        #print('facessss', faces)

        for (x, y, w, h) in faces:
            image = cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imwrite("/home/mervek/Desktop/bitirme/frame_/withoutcrop/%d.jpg" % (count), image)

            #crop_img = image[y:y + h, x:x + w]
            #k_image = cv2.resize(crop_img, (64, 64))
            #cv2.imwrite("/home/mervek/Desktop/bitirme/frame_/%d.jpg" %(count), k_image)

        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
    return count


def tocrope(path,framesayisi,topath):    #kırpılacak resimlerin oldugu dosya yolunu ve içersindeki frame sayısını verip her resimdeki yüzü yakalayıp kırpma ve yeni path adresine kaydetmekaydetme
    n=0
    while n!=framesayisi:
        img=cv2.imread("%s/%d.jpg" %(path,n))
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in faces:
            image=cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            crop_img = image[y:y + h, x:x + w]
            cv2.imwrite("%s/%d.jpg" % (topath,n), crop_img)
        n+=1


def resize(path,framesayisi,topath):    #kırpılacak resimlerin oldugu dosya yolunu ve içersindeki frame sayısını verip her resmi kırp
    n = 0
    while n != framesayisi:
        img = cv2.imread("%s/%d.jpg" % (path, n))
        r_img=cv2.resize(img,(224,224))
        cv2.imwrite("%s/%d.jpg" % (topath,n), r_img)
    n += 1

def hogla(path,framesayısı):
    winSize = (64, 64)        #resimleri64*64yaptım,16*16lık bloklarla(yani8*8likhucreler)yani yatayda ve dikeyde 8er tane blok var yataydaki her blokta 7kezgezerdikeydede7keztoplmda49gezme her bolkta 8*8lik 4hucre var sonucta 9histogram*4lükhücreden sonucta 36 deger var 49gezme*36 degerden 1764 veri elde ederiz
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                            histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    n=1    #mmı 1den baslatmıs bizimkiler 0
    mat = []
    while n!=framesayısı:
        img = cv2.imread("%s/%d.jpg" % (path, n))
        r_img=cv2.resize(img, (64, 64))
        mat.append(hog.compute(r_img))
        x = numpy.asarray(mat)
        n+=1

    print(len(x[0]))
    return x

def fonksiyonlar(x,csvisim) : #hoglanmış matrisi buraya yolla, yani  x yerine bunu =>hogla(,) yazabiliriz diğerinede csvyi kaydedicegimiz adını yazabiliriz yazabiliriz
    with open("/home/mervek/Desktop/bitirme/csv/%s.csv" %csvisim, "a") as path:
        wr = csv.writer(path)
        n=0
        while n!=99:
            wr.writerow(x[n])
            n+=1




#videofrompathtoframe("output.avi")
#withoutcrope("emotionvideo.avi")
#videofromwebcamtoframe(7)

#tocrope("/home/mervek/Desktop/bitirme/frame_/withoutcrop",100,"/home/mervek/Desktop/bitirme/frame_/withcrop")
#resize("/home/mervek/Desktop/bitirme/frame_/withcrop",100,"/home/mervek/Desktop/bitirme/frame_/resize")
#hogla("/home/mervek/Desktop/CKP/Untitled Folder/MMI/MMI_DPM_alignment/100",83)
#hogla("/home/mervek/Desktop/CKP/Untitled Folder/MMI/MMI_DPM_alignment/101",99)
#def polifiticinvideodakibutunresimlerinozniteliklerinincsvsi():




