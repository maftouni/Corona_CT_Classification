import tensorflow as tf
import keras
config =  tf.compat.v1.ConfigProto( device_count = {'GPU': 4 } ) 
sess = tf.compat.v1.Session(config=config) 
from tensorflow.compat.v1.keras import backend as K
K.set_session(sess)


import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model,Sequential
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import PIL



def estimate(X_train,y_train):
    i = 0
    nrows=224
    ncolumns=224
    channels=1
    ntrain=0.9*len(X_train)
    nval=0.1*len(X_train)
    batch_size=16
    epochs=100
    
    X = []
    X_train=np.reshape(np.array(X_train),[len(X_train),])
    
    for img in list(range(0,len(X_train))):
        if X_train[img][0].ndim>=3:
            X.append(cv2.resize(X_train[img][:,:,:3], (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
        else:
            smimg= cv2.cvtColor(X_train[img][0],cv2.COLOR_GRAY2RGB)
            X.append(cv2.resize(smimg, (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
        
        if y_train[img]=='COVID':
            y_train[img]=1
        elif y_train[img]=='NonCOVID' :
            y_train[img]=0
        else:
            continue

    x = np.array(X)
    X_train, X_val, y_train, y_val = train_test_split(x, y_train, test_size=0.10, random_state=2)
    
    train_datagen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    model = models.Sequential()
    model.add(tf.keras.applications.DenseNet169(include_top=False, input_shape = (224, 224, 3) , weights='imagenet',pooling= 'avg'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
      
    
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator= train_datagen.flow(X_val, y_val, batch_size=batch_size)
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=ntrain//batch_size,epochs=epochs,
                                  validation_data=val_generator,
                                  validation_steps=nval // batch_size)
    model.save("Model.h5")
    return model

def predict(X_test,model):
    i = 0
    nrows=224
    ncolumns=224
    channels=1
 
    X = []
    X_test=np.reshape(np.array(X_test),[len(X_test),])
    
    for img in list(range(0,len(X_test))):
        if X_test[img][0].ndim>=3:
            X.append(cv2.resize(X_test[img][:,:,:3], (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
        else:
            smimg= cv2.cvtColor(X_test[img][0],cv2.COLOR_GRAY2RGB)
            X.append(cv2.resize(smimg, (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
    x = np.array(X)
    
    y_pred=[]
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    for batch in test_datagen.flow(x, batch_size=1):
        pred = model.predict(batch)
        if pred > 0.5:
            y_pred.append('COVID')
            #print('covid')
        else:
            y_pred.append('NonCOVID')
            #print('noncovid')
        i+=1
        if i%len(X_test)==0:
            break
            
    return y_pred



dbfile = open('training.pickle', 'rb')      
db = pickle.load(dbfile) 
print(len(np.array(db['y_tr'])))


model = estimate(db['X_tr'],db['y_tr'])

dbfile = open('test.pickle', 'rb')      
db_test = pickle.load(dbfile) 

y_pred = predict(db_test['X_tr'],model)

