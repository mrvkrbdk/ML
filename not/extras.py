
dataset = pd.read_csv("/home/mervek/Desktop/etiketvedosya.csv")
array = dataset.values
X = array[:, 0:]
ee = numpy.zeros(208)
ee[1:len(ee)+1] = numpy.squeeze(X)
print(ee[0])
print(ee[1])
print(ee[206])
print(ee[207])

dataset = pd.read_csv("/home/mervek/Desktop/etiketvedosya.csv")
array2 = dataset.values
print(dataset.shape)
Y = array2[:, 0:]
ee = numpy.zeros(207)
ee[0:len(ee)] = numpy.squeeze(Y)
print(ee)
dataset = pd.read_csv("/home/mervek/Desktop/bitirme/csv/mmıdata.csv")
print(dataset.shape)
dataset = pd.read_csv("/home/mervek/Desktop/bitirme/csv/hogmmiwithpolyfitparameter1.csv")
print(dataset.shape)
array = dataset.values
aa=array[:,0]
print(aa)
X = array[:,8820]
print(X)




#path="/home/mervek/Desktop/CKP/Untitled Folder/MMI/MMI_DPM_alignment/"
#li=os.listdir(path)
#ar2=numpy.asarray(li)
#ar=sorted(ar2)
#len(os.listdir(path+os.listdir(path)[12]))
#print(ar2)
#print(ar[12])
#print(len(os.listdir(path+ar[12])))
#print(os.listdir("/home/mervek/Desktop/CKP/Untitled Folder/MMI/MMI_DPM_alignment/")[174])
#print(len(os.listdir("/home/mervek/Desktop/CKP/Untitled Folder/MMI/MMI_DPM_alignment/"+os.listdir("/home/mervek/Desktop/CKP/Untitled Folder/MMI/MMI_DPM_alignment/")[0])))
#print(len(os.listdir("/home/mervek/Desktop/CKP/Untitled Folder/MMI/MMI_DPM_alignment/")))









#X = array[:,1:]
#with open("/home/mervek/Desktop/bitirme/csv/mmıdata.csv", "a") as fp:
 #   wr = csv.writer(fp)
  #  n=0
   # while n!=234:
    #    wr.writerow(X[:,n])
     #   n=+1



#print(X2)
#dataset=pd.read_csv("/home/mervek/Desktop/bitirme/csv/mmıdata.csv")                --
#dataset.shape
#Y = array[:,8820]                                                                  --
#Y2 = array[:,8821]
#print(dataset.shape)
#print(Y)                                                                     --
