# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 22:44:14 2020

@author: aycaburcu
"""

from keras.datasets import mnist
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical


#x_train bizim giriş veri setimiz.
#x_train içinde görüntüler var y_train içinde ise hedefler bulunmaktadır.

(x_train,y_train),(x_test,y_test)=mnist.load_data()
#mnist veri setini yüklemeliyiz:
#x_train bizim giriş veri setimiz.
#x_train içinde görüntüler var y_train içinde ise hedefler bulunmaktadır.


#Normalizasyon işlemleri:
x_train=x_train.reshape((60000,28*28))#Goruntuyu vektore donusturur
x_train=x_train.astype('float32')/255#Normalize eder
x_test=x_test.reshape((10000,28*28))
x_test=x_test.astype('float32')/255

print("Normalizasyon sonrasi shape:",x_train.shape)
#modeli tanımlamalıyız:
model=models.Sequential()



#modelin yapısını tanımlamalıyız:
#giris katmanı
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(16,activation='relu',input_shape=(28*28,)))
#burada 28*28 yerine 784 yazılabilir

#buraya bir katman daha ekleyedik.
model.add(layers.Dense(16,activation='relu'))
"""Çıkış katmanımızda 10 adet çıkış olmalı.  
0,1,2,3,4,5,6,7,8,9
Bu, **multi class classification** olarak geçer. Dolayısıyla çıkış katmanımızda activation function olarak **softmax** tercih ediyoruz.
**Binary Classification**'da ise genelde **sigmoid** tercih edilir.
"""
model.add(layers.Dense(10,activation='softmax'))

#stochastic gradient descent kullandık.
##multi classification olduğu için categorical cross entropy kullandık.
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




"""`categorical_crossentropy` kullanabilmemiz için, çıkış datalarımız olan *y_train* ve *y_test* datalarını categorical'a dönüştürmemiz gerekiyor.
Örneğin, **5** olarak değil **[0,0,0,0,0,1,0,0,0,0]** olarak modele vermemiz gerekiyor.
`sparse_categorical_crossentropy` bu işlemi kendisi yapar. `to_categorical` yapmamıza gerek kalmaz.
"""
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

y_train[0]

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


y_train.shape

y_train[5,:]
#eğittiğimiz ağın performansını kontrol edeceğiz şimdi:
model.evaluate(x_test,y_test)
#bu bize loss ve acc döndürdü

#oluşturulan modeli incelemek iç
model.summary()
#girişimiz kaç elemanlı=784, kaç hücre var=16 
#her hücreden girişteki elemanlara bağlantı var bu bağlantıların her birinde weight var
#784x16=12544 16 tanede her hücre için bias var 12544+16=12560
#784*16+16
#ikinci katmanın giriş sayısı bir önceki katmanın çıkış sayısı
#16x16+16
#son katmanda da 16x10+10=170
total_params=(784*16+16)+(16*16+16)+(16*10+10)
total_params

### Train
#Sıradaki adımımız modeli eğitmek.

model.fit(x_train,y_train,epochs=100,batch_size=64)


test_loss,test_acc=model.evaluate(x_test,y_test)
test_loss

w=model.get_weights()
w
#ben bu weightleri kaydedersem daha sonradan kullanabilirim.
model.save('mnist.h5')


model.summary()
model.save_weights('weightsmnist.h5')
model=models.load_model('mnist.h5')
model.summary()







