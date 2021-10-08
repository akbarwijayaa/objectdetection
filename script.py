import numpy as np
import cv2
import matplotlib.pyplot as plt

cascade = cv2.CascadeClassifier('cascade.xml')
img = cv2.imread('object6.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pipe = cascade.detectMultiScale(
    gray, 
    scaleFactor=1.02,
    minNeighbors = 4, 
    minSize=(100, 100), 
    flags = cv2.CASCADE_SCALE_IMAGE)

for(x,y,w,h) in pipe :
    center_point = [(x+(w/2)), (y+(h/2))]
    radius = int(w/5)
    img2 = cv2.circle(img, (int(center_point[0]), int(center_point[1])), radius, (255, 0, 0), -1)
    
# Menjumlahkan object yang terdeteksi
print("Jumlah Object yang terdeteksi adalah {0} pipa!".format(len(pipe)))

plt.imshow(img)
plt.show()

# #Mencari False Positive
# a = len(pipe)
# print("Jumlah yang terdeteksi: " , a)
# b = 61
# tambah = int(input("nilai false negative: "))

# c = a - b + tambah
# print("Jumlah False Positifnya adalah : ", c)


# a = 140 # jumlah seluruh pipa yang harus dihitung
# b = len(pipe) # Menjumlahkan pipa yang dihitung random oleh komputer
# c = a - b # jumlah seluruh pipa yang harus dihitung dikurangi hasil dari komputer
# d = c - int(input("double :")) # karena masih ada double, maka dikurangi double 
# print("False Positif terdapat {0} pipa".format(d))
