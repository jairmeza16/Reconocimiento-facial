import tensorflow as tf
#import tensorflow_datasets as tfds
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop
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
#Hay que eliminar el doble espacio entre los datos, por
#eso está fallando al cargarlos. 
'''with open('list_attr_celeba.txt', 'r') as f:
    print("skipping : " + f.readline())
    print("skipping headers : " + f.readline())
    with open('attr_celeba_prepared.txt' , 'w') as newf:
        for line in f:
            new_line = ' '.join(line.split())
            newf.write(new_line)
            newf.write('\n')'''

df = pd.read_csv('attr_celeba_prepared.txt', sep=' ', header = None)
#Cambiando los -1 por 0
df= df.replace([-1],0)
print('----------')
#print(df[0].head())
files = tf.data.Dataset.from_tensor_slices(df[0])
#print(f'esto es files {files}')
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
#print(f'esto es attributes {attributes}')
data = tf.data.Dataset.zip((files, attributes))
#print(f'esto es data {data}')

path_to_images = 'C:/Users/jairm/OneDrive/Documentos/Redes neuronales/Reconocimiento_facial/img_align_celeba/img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

labeled_images = data.map(process_file)

'''print('AQUÍ IMPRIMÍ ALGO')
print(labeled_images)'''
#print('Aquí imprimí la dirección de las imágenes')
#print(path_to_images+'0001.jpeg')

'''print('AQUÍ IMPRIMÍ ALGUNAS IMÁGENES')
for image, attri in labeled_images.take(2):
    plt.imshow(image)
    plt.show()
'''
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
'''print(type(train_ds))
print("Aquí imprimí cosas 2")
train_ds.__len__()//32 #steps per epoch
print(train_ds.__len__()) #train_ds tiene 162079 imágenes
print(val_ds.__len__()) #val_ds tiene 20259 imágenes'''
tstep=162079//32 #5064
vstep=20259//32 #633



#Aquí estoy especificando el batch para ambos datasets Igual le puse que se repitiera el número de veces de las épocas, la primera vez que lo corrí y que no repetí los datos
# me dijo que se me habían acabado los datos y que debía repetir el dataset...
train_ds = train_ds.batch(32).repeat(8)
val_ds = val_ds.batch(32).repeat(8)

#Escribiendo la red 
model = Sequential()
#Cambié el input_shape, pero no sé si hay que ponerle el tamaño que tenían las imágenes o algo del tamaño del tensor...
model.add(Conv2D(10, (3, 3), input_shape=(192,192,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
#Aquí creo que hay que cambiar el número de neuronas en la capa de salida. Tenía 1, ahora serían 40 atributos. 
model.add(Dense(40))
#Cambiar a tangente hiperbólica. (Ya no es necesario cambiarlo porque ya tengo en el tensor solamente 0 y 1.)
model.add(Activation('sigmoid'))
#Dijo que no iba a jalar la binary_crossentropy porque tengo valores negativos en los datos y tengo un logaritmo. Entonces parece que lo más sencillo es cambiar
#el archivo de los atributos, todos los -1 cambiarlos a 0.
opt = keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
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


model.save("red.h5")







