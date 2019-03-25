from keras.datasets import *
from keras.models import *
from math import *
from keras.utils import *
import skimage.io as io
import skimage.data as dt
import skimage.transform as tra
from keras.engine import *
from keras.wrappers import *
import numpy as np
import scipy.io as sio
import h5py
from keras.backend import *
from keras.layers import *
import tensorflow as tf

from keras.optimizers import *


def d_loss(y_true,y_pred):
    return 2*mean((0.5 - y_true)*y_pred)

def g_loss(y_true,y_pred):
    return -d_loss(y_true,y_pred)

def find_mean(A):
    S = A[0,:]
    for i in range(1,A.shape[0]):
        S = S+ A[i,:]
    return S/A.shape[0]

P = 0.9
size = 512
x = 0
y = []

def fun(num,day,part):
    file = 'C:/Users/myfamily/Documents/Tencent Files/2408409697/FileRecv/MobileFile/Newdata/Newdata/YXday10_part1.mat'
    data = h5py.File(file)
    train_set_data = data['trainData'][:]
    x1 = np.array(train_set_data)
    x = x1[155*(num-1):155*num, :16]
    y = x1[155*(num-1):155*num, 16]
    source_data_x = x
    source_data_y = y
    file = 'C:/Users/myfamily/Documents/Tencent Files/2408409697/FileRecv/MobileFile/Newdata/Newdata/YXday10_part2.mat'
    data = h5py.File(file)
    train_set_data = data['trainData'][:]
    x1 = np.array(train_set_data)
    x = x1[155*(num-1):155*num, :16]
    y = x1[155*(num-1):155*num, 16]
    source_data_x = np.concatenate((x, source_data_x))
    source_data_y = np.concatenate((y, source_data_y))

    for n in range(1, 10):
        i = n
        j = 1
        file = 'C:/Users/myfamily/Documents/Tencent Files/2408409697/FileRecv/MobileFile/Newdata/Newdata/YXday0' + str(
            i) + '_part' + str(j) + '.mat'
        data = h5py.File(file)
        train_set_data = data['trainData'][:]
        x1 = np.array(train_set_data)
        x = x1[155*(num-1):155*num, :16]
        y = x1[155*(num-1):155*num, 16]
        source_data_x = np.concatenate((x, source_data_x))
        source_data_y = np.concatenate((y, source_data_y))

    for n in range(1, 10):
        i = n
        j = 2
        file = 'C:/Users/myfamily/Documents/Tencent Files/2408409697/FileRecv/MobileFile/Newdata/Newdata/YXday0' + str(
            i) + '_part' + str(j) + '.mat'
        data = h5py.File(file)
        train_set_data = data['trainData'][:]
        x1 = np.array(train_set_data)
        x = x1[:155, :16]
        y = x1[:155, 16]
        source_data_x = np.concatenate((x, source_data_x))
        source_data_y = np.concatenate((y, source_data_y))

    source_data_x = np.array(source_data_x)

    #source_data_x = normalize(source_data_x)
    for i in range(len(source_data_x)):
        source_data_x[i,:] = (source_data_x[i,:] - np.min(source_data_x[i,:]))/(np.max(source_data_x[i,:]) - np.min(source_data_x[i,:]))
    print(source_data_x[1, :])
    print(source_data_x.shape)

    file = 'C:/Users/myfamily/Documents/Tencent Files/2408409697/FileRecv/MobileFile/Newdata/Newdata/BLday0'+str(day)+'_part'+str(part)+'.mat'
    data = h5py.File(file)
    train_set_data = data['trainData'][:]

    x1 = np.array(train_set_data)
    target_data_x = x1[155*(num-1):155*num, :16]
    target_data_x = np.array(target_data_x)
    for i in range(len(target_data_x)):
        target_data_x[i,:] = (target_data_x[i,:] - np.min(target_data_x[i,:]))/(np.max(target_data_x[i,:]) - np.min(target_data_x[i,:]))
    print(target_data_x[1, :])

    k = target_data_x
    for i in range(19):
        target_data_x = np.concatenate((target_data_x, k))
    print(target_data_x.shape)

    G = Sequential()
    G.add(Dense(16, input_dim=16, activation='relu'))

    G.add(Dense(16, activation='relu'))

    G.add(Dense(16, activation='relu'))

    G.add(Dense(16, activation='sigmoid'))
    G.summary()

    D = Sequential()
    D.add(Dense(16, activation='relu', input_dim=16))

    D.add((Dense(16, activation='relu')))

    D.add((Dense(16, activation='relu')))

    D.add(Dense(1,activation='sigmoid'))
    D.summary()

    DM = Sequential()
    DM.add(D)
    DM.compile(loss=d_loss, optimizer='RMSprop', metrics=['accuracy'])

    AM = Sequential()
    AM.add(G)
    AM.add(D)
    AM.compile(loss=g_loss, optimizer='RMSprop', metrics=['accuracy'])

    GM = Sequential()
    GM.add(G)
    GM.compile(loss='mse', optimizer='RMSprop', metrics=['accuracy'])

    number = int(len(source_data_x) * P)
    j = target_data_x
    index = np.random.choice(len(source_data_x), number, replace=False)

    print(index)
    D_train = source_data_x[index]
    G_train = target_data_x[index]

    D_test = np.delete(source_data_x, index, axis=0)
    G_test = np.delete(target_data_x, index, axis=0)

    print(len(D_train))
    print(len(D_test))
    print(len(source_data_x))

    for i in range(2000):
        print(i)
        X_train1 = G_train[np.random.randint(0, G_train.shape[0], size=size), :]
        X_train2 = D_train[np.random.randint(0, D_train.shape[0], size=size), :]
        fake = GM.predict(X_train2)
        print(fake)

        train_trueAndFake = np.concatenate((fake, X_train1))
        result = np.ones((size * 2, 1))
        result[:size, :] = 0

        DM.fit(train_trueAndFake, result, epochs=1, batch_size=size)


        z = DM.predict(train_trueAndFake)
        for l in D.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -10, 10) for w in weights]
            l.set_weights(weights)
        print(z[0])
        print(z[-1])
        result = np.ones((size, 1))
        AM.fit(X_train2, result, epochs=1, batch_size=size)

    k = G.predict(D_test)
    print(k.shape)
    y = [num]*len(k)
    y = np.array(y).T

    print(len(k))
    print(len(G_test))
    print(find_mean(k))
    print(find_mean(G_test))
    return [k,y]

for day in range(1,10):
    for part in range(1,3):
        data = fun(1,day,part)
        x = data[0]
        y = data[1]
        for i in range(1,10):
            data = fun(i+1,day,part)
            x = np.concatenate((x, data[0]))
            y = np.concatenate((y, data[1]))

        file = 'C:\\Users\\myfamily\\Desktop\\新建文件夹 (2)\\BL_0'+str(day)+'_G_'+str(part)+'.mat'

        sio.savemat(file,{'X':x,'Y':y})
