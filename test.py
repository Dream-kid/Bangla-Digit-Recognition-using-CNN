# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:05:49 2020

@author: lenovo
"""
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

CATEGORIES=["0","1","2","3","4","5","6","7","8","9"]

def prepare(filepath):
    IMG_SIZE=28
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_array, cmap='gray')
    plt.show()
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



model=tf.keras.models.load_model("64x3-CNN.model")
predection=model.predict([prepare('input/1.tif')])

print(predection)
for i in range(10):
    temp = int(predection[0][i])
    if temp ==1:
        print(i)
        break
#print(CATEGORIES[int(predection[0][0])])

