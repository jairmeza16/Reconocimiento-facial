import tensorflow as tf
import pathlib
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
print('----------')
#print(df[0].head())
files = tf.data.Dataset.from_tensor_slices(df[0])
#print(f'esto es files {files}')
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
#print(f'esto es attributes {attributes}')
data = tf.data.Dataset.zip((files, attributes))
print(f'esto es data {data}')

path_to_images = 'C:/Users/jairm/OneDrive/Documentos/Redes neuronales/Reconocimiento_facial/img_align_celeba/img_align_celeba'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

labeled_images = data.map(process_file)

print(labeled_images)

'''for image, attri in labeled_images.take(1):
    plt.imshow(image)
    plt.show()'''

#Separando imágenes en subconjuntos de datos. 
#shuffled_dataset=labeled_images.shuffle(10) -> Este fue el intento que hice yo para bajarar los datos, sin embargo encontré una función que los baraja
#y de una vez divide el dataset en 3 subconjuntos. 
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    
    return train_ds, val_ds, test_ds
print('--------------')
print(labeled_images.__len__()) #Para saber la longitud del dataset.

#En la siguiente línea tuve problemas, pues pensé que al llamar la función, me iba a devolver las variables train_ds, val_ds, test_ds, pero en realidad devuelve los 
#valores... Entonces hay que crear las variables e igualarlas a los valores que devuelve la función para poder llamar a estas variables en la red neuronal.
train_ds, val_ds, test_ds = get_dataset_partitions_tf(labeled_images, 202599)






