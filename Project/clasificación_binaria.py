import os
from glob import glob
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
#pip install sklearn
#pip install scikit-image
from sklearn.model_selection import train_test_split
#from skimage import color, io
#from scipy.misc import imresize  ## deprecated
from PIL import Image
trainme = 'data/mini_train/me/'
trainotme = 'data/mini_train/notme/'
me_files_path = os.path.join(trainme, '*')
notme_files_path = os.path.join(trainotme, '*')
me_files = sorted(glob(me_files_path))
notme_files = sorted(glob(notme_files_path))
n_files = len(me_files) + len(notme_files)
print(n_files)

size_image = 192
allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0

#Todos los me_files ya tienen el tamaño deseado, pues los puse a mano. Pienso que es bueno saltarme esta parte del código.
'''for f in me_files:
    try:
        #img = io.imread(f)
        #new_img = imresize(img, (size_image, size_image, 3))
        img = Image.open(f)
        new_img = img.resize(size=(size_image, size_image))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
        print("Imagen cargada")
    except:
        print("No cargo imagen")
'''        #continue
for f in notme_files:
    try:
        img = Image.open(f)
        new_img = img.resize(size=(size_image, size_image))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue
x, x_test, y, y_test = train_test_split(allX, ally, test_size=0.2, random_state=1)

#print(x.shape[0])
#print(x_test.shape)
#print(y.shape)
#print(y_test.shape)
x_train = x.reshape(x.shape[0], 192*192*3)
x_test = x_test.reshape(x_test.shape[0], 192*192*3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.
x_train.shape
x_train.shape
#y_train = to_categorical(y, 2)
#y_test = to_categorical(y_test, 2)
y_train = y
#print(y_train)
#print(y_train.shape)
#print("-----")
#print(x_train.shape)

#y_test = y_test
#Aquí voy a añadir el modelo que ya entrené, congelaré las primeras capas y solamente añadiré alguna convolucional con una densa de una neurona, para que pueda
#diferenciar entre mí y las fotos en las que no aparezco.
'''pre_trained_model=load_model('red3.h5')
model = tf.keras.Sequential()
model.add(pre_trained_model.layers[4])
model.add(Dense(1, activation='sigmoid'))
for layer in model.layers[:11]:
    layer.trainable = False
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history = model.fit(x_train, y_train,batch_size=128,epochs=30,verbose=1,validation_data=(x_test, y_test))
model.summary()
score = model.evaluate(x_test, y_test, verbose=0)
print(score)'''
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(192*192*3,)))
#model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history = model.fit(x_train, y_train,batch_size=128,epochs=30,verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print(score)