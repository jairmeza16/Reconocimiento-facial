import tensorflow as tf
#import tensorflow_datasets as tfds
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop
import pathlib
import datetime
from tensorflow.keras.models import load_model
import os
from turtle import pd

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Ahora vamos a cargar imágenes que no son de mis rostro que no sean de CelebA
#Obtengamos una lista de los nombres de las imágenes en la carpeta.
files_names_notme = os.listdir('C:/Users/jairm/OneDrive/Documentos/Repositorios/Reconocimiento_facial/Project/data/train/notme_new')
attributes_notme = [0] * 5645
df_notme = pd.DataFrame((zip(files_names_notme, attributes_notme)))
print(df_notme.shape)
#Ahora, hagamos el dataset...


files_notme = tf.data.Dataset.from_tensor_slices(df_notme[0])
#print(f'esto es files {files}')
attributes_notme = tf.data.Dataset.from_tensor_slices(df_notme.iloc[:,1:].to_numpy())
#print(f'esto es attributes {attributes}')
data_notme = tf.data.Dataset.zip((files_notme, attributes_notme))
#Para identificar a data y su forma:
print(f'esto es data {data_notme}')
print(data_notme.__len__())
#Ahora, añadimos las imágenes...


path_to_images = 'C:/Users/jairm/OneDrive/Documentos/Repositorios/Reconocimiento_facial/Project/data/train/notme_new/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

labeled_images_notme = data_notme.map(process_file)

print('AQUÍ IMPRIMÍ ALGO')
print(labeled_images_notme)

#Para corroborar que se añadieron correctamente, vamos a imprimir algunas.
print('AQUÍ IMPRIMÍ ALGUNAS IMÁGENES')
for image, attri in labeled_images_notme.take(2):
    plt.imshow(image)
    plt.show()



#Hagamos lo mismo para las imágenes que son de mi rostro...
#Obtengamos una lista de los nombres de las imágenes en la carpeta.
files_names_me = os.listdir('C:/Users/jairm/OneDrive/Documentos/Repositorios/Reconocimiento_facial/Project/data/train/me')
attributes_me = [1] * 7732
df_me = pd.DataFrame((zip(files_names_me, attributes_me)))
print(df_me.shape)
#Ahora, hagamos el dataset...


files_me = tf.data.Dataset.from_tensor_slices(df_me[0])
#print(f'esto es files {files}')
attributes_me = tf.data.Dataset.from_tensor_slices(df_me.iloc[:,1:].to_numpy())
#print(f'esto es attributes {attributes}')
data_me = tf.data.Dataset.zip((files_me, attributes_me))
#Para identificar a data y su forma:
print(f'esto es data {data_me}')
print(data_me.__len__())
#Ahora, añadimos las imágenes...


path_to_images = 'C:/Users/jairm/OneDrive/Documentos/Repositorios/Reconocimiento_facial/Project/data/train/me/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

labeled_images_me = data_me.map(process_file)

print('AQUÍ IMPRIMÍ ALGO')
print(labeled_images_me)

#Para corroborar que se añadieron correctamente, vamos a imprimir algunas.
'''print('AQUÍ IMPRIMÍ ALGUNAS IMÁGENES')
for image, attri in labeled_images_me.take(2):
    plt.imshow(image)
    plt.show()'''


concat_ds = labeled_images_me.concatenate(labeled_images_notme)
print(labeled_images_notme.__len__()) #Tamaño del dataset es 5645
print(labeled_images_me.__len__()) #Tamaño del dataset es 7732
print(concat_ds.__len__()) #Tamaño del dataset es la suma de los dos anteriores, es decir, 13377.

#Ahora, vamos a separar el dataset en 3, entrenamiento, test y validación. También vamos a revolverlos. Reciclaremos el código que usamos al cargar los
#atributos de CelebA.

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=22000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        #Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
        print('the dataset has been shuffled')
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(concat_ds, 13377)
print(type(train_ds))
print("Aquí imprimí cosas 2")

print(train_ds.__len__()) #train_ds tiene 10701 imágenes
print(val_ds.__len__()) #val_ds tiene 1337 imágenes

tstep=10701//32 #334
vstep=1337//32 #41
#Aquí estoy especificando el batch para ambos datasets Igual le puse que se repitiera el número de veces de las épocas, la primera vez que lo corrí y que no repetí los datos
# me dijo que se me habían acabado los datos y que debía repetir el dataset...
train_ds = train_ds.batch(32).repeat(8)
val_ds = val_ds.batch(32).repeat(8)
test_ds = test_ds.batch(32).repeat(8)

#Ahora, construyamos propiamente la red neuronal con las capas de la anterior que fue entrenada. Para ello, hay que cargar el último modelo entrenado.
pre_trained_model=load_model('red5_fotos_amigos.h5')
model = tf.keras.Sequential()
model.add(pre_trained_model.layers[0])
model.add(pre_trained_model.layers[1])
model.add(pre_trained_model.layers[2])
model.add(pre_trained_model.layers[3])
model.add(pre_trained_model.layers[4])
model.add(pre_trained_model.layers[5])
model.add(pre_trained_model.layers[6])
model.add(pre_trained_model.layers[7])
model.add(pre_trained_model.layers[8])
model.add(pre_trained_model.layers[9])
model.add(pre_trained_model.layers[10]) 
#model.add(pre_trained_model.layers[11])

for layer in model.layers[:10]:
    layer.trainable = False
model.summary()
opt = keras.optimizers.RMSprop(learning_rate=0.0001)
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
                epochs=8,
                validation_data=val_ds,
                validation_steps=vstep,
                #callbacks=[tbCallBack]
                )


model.save("red5_fotos_amigos.h5")
#Estoy intentando evaluar el modelo con el dataset de evaluación... añadí captura de pantalla en la carpeta de fotos_entrenamiento
model = load_model('red5_fotos_amigos.h5')
loss, accuracy = model.evaluate(test_ds)
print(loss)
print(accuracy)
