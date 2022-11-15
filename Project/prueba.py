import tensorflow as tf
#import tensorflow_datasets as tfds
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
import pathlib
import datetime
import os
from turtle import pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#Aquí cargamos el modelo para seguir entrenando. Creo que hay que poner otra vez el set de datos... Lo pondré a continuación.
df = pd.read_csv('attr_celeba_prepared.txt', sep=' ', header = None)
#Cambiando los -1 por 0
df= df.replace([-1],0)
print('----------')
files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files, attributes))
path_to_images = 'C:/Users/jairm/OneDrive/Documentos/Redes neuronales/Reconocimiento_facial/img_align_celeba/img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

labeled_images = data.map(process_file)

#Separando imágenes en subconjuntos de datos. 
#shuffled_dataset=labeled_images.shuffle(10) -> Este fue el intento que hice yo para bajarar los datos, sin embargo encontré una función que los baraja
#y de una vez divide el dataset en 3 subconjuntos. 
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1):#, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    #if shuffle:
        # Specify seed to always have the same split distribution between runs
        #ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    
    return train_ds, val_ds, test_ds
#print('--------------')
#print(labeled_images.__len__()) #Para saber la longitud del dataset.

#En la siguiente línea tuve problemas, pues pensé que al llamar la función, me iba a devolver las variables train_ds, val_ds, test_ds, pero en realidad devuelve los 
#valores... Entonces hay que crear las variables e igualarlas a los valores que devuelve la función para poder llamar a estas variables en la red neuronal.
train_ds, val_ds, test_ds = get_dataset_partitions_tf(labeled_images, 202599)

tstep=162079//32 #5064
vstep=20259//32 #633

train_ds = train_ds.batch(32).repeat(8)
val_ds = val_ds.batch(32).repeat(8)

model = load_model('red3.h5')


opt = keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#Voy a comentar la siguiente línea por el momento. Es cuando se le pide a tensorboard que haga algo...
#tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

#python -m tensorboard.main --logdir=/Graph  <- Para correr Tensor board
#tensorboard  --logdir Graph/
print("Logs:")
print(log_dir)
print("__________")



model.fit(
                train_ds,
                steps_per_epoch=tstep,
                epochs=2,
                validation_data=val_ds,
                validation_steps=vstep,
                #callbacks=[tbCallBack]
                )


model.save("red3.h5")