#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 20:47:59 2020

@author: aycaburcu
"""
from keras.datasets import mnist
from keras import models

import matplotlib.pyplot as plt
import numpy as np

(x_train,y_train),(x_test,y_test)=mnist.load_data()
model=models.load_model('mnist.h5')




inx=np.random.randint(0,10000)
rakam=x_test[inx,:,:]



#seçilen görüntüyü modele uygula

y=model.predict(rakam.reshape(1,784)/255)
tahmin_sonuc=np.argmax(y)

ax=plt.subplot()
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlabel('label:{0}\npred:{1}'.format(y_test[inx],tahmin_sonuc))
plt.imshow(rakam,cmap='gray')






