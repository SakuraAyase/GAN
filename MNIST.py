from keras.datasets import *
from keras.models import *
import skimage.io as io
import skimage.data as dt
import skimage.transform as tra
from keras.engine import *
from keras import *
import numpy as np
from io import *

from keras.backend import *
from keras.layers import *
import tensorflow as tf
import PIL.Image as img

from keras.optimizers import *

from tensorflow.examples.tutorials.mnist import input_data


def plot(image,i):
    image = np.reshape(image,[28,28])
    fileName = 'C:\\Users\\myfamily\\Desktop\\新建文件夹2\\'
    io.imsave(fileName+'lean'+str(i)+'.jpg',image)
x_train = input_data.read_data_sets("mnist",\
        	one_hot=True).train.images

x_train = 1-x_train;
plot(x_train[0],1)
x_train = x_train.reshape(-1,784).astype(np.float32)

y_train = input_data.read_data_sets("mnist",one_hot=True).train.labels
print(y_train.shape)
print(x_train.shape)


D = Sequential()

D.add(Dense(64,input_dim=784,activation='sigmoid'))

D.add(Dense(64,activation='sigmoid'))
D.add(Dense(10,activation='softmax'))

D.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])

D.fit(x_train,y_train,epochs=10)

file = open("thefile.txt",'w')
temp = D.layers[0].get_weights()[0]
print(temp.shape)
for i in range(temp.shape[0]):
    file.write(str(list(temp[i,:])))
file.close()
file = open("thefile1.txt",'w')
temp = D.layers[1].get_weights()[0]
print(temp.shape)
for i in range(temp.shape[0]):
    file.write(str(list(temp[i,:])))

file.close()

file = open("thefile2.txt",'w')
temp = D.layers[2].get_weights()[0]
print(temp.shape)
for i in range(temp.shape[0]):
    file.write(str(list(temp[i,:])))
file.close()
"""
temp = D.layers[1].get_weights()[0]
print(temp.shape)
for i in range(temp.shape[0]):
    file.write(str(list(temp[i,:])))
file.write('ddsd')

temp = D.layers[2].get_weights()[0]
print(temp.shape)
for i in range(temp.shape[0]):
    file.write(str(list(temp[i,:])))
"""

