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

#Vamos a intentar cargar algunas fotos que la red no ha visto para ver su predicción.
#Para ello haré un dataset que contenga una foto. Pondré su etiqueta para que tenga la misma dimensión que requiere la red, sin embargo, el
#model.predict() solamente me dará la predicción de la foto, sin comparar con la etiqueta.
model = load_model('red4.h5')
files_names_me = ['pendiente24.png']
attributes_me = [0]
df_me = pd.DataFrame((zip(files_names_me, attributes_me)))
files_me = tf.data.Dataset.from_tensor_slices(df_me[0])
#print(f'esto es files {files}')
attributes_me = tf.data.Dataset.from_tensor_slices(df_me.iloc[:,1:].to_numpy())
#print(f'esto es attributes {attributes}')
data_me = tf.data.Dataset.zip((files_me, attributes_me))

#path_to_images = 'C:/Users/jairm/OneDrive/Documentos/Redes_neuronales/Reconocimiento_facial/img_align_celeba/img_align_celeba/'
path_to_images = 'C:/Users/jairm/OneDrive/Documentos/Repositorios/Reconocimiento_facial/Project/data/base_fotos_propias/'

def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    return image, attributes

labeled_images_me = data_me.map(process_file)
for image, attri in labeled_images_me.take(1):
    plt.imshow(image)
    plt.show()
labeled_images_me = labeled_images_me.batch(1)
prediction = model.predict(labeled_images_me)
print(prediction)
if prediction<=0.5:
    prediction = 'no es mía'
else:
    prediction = 'es mía'
print(f"Esta foto probablemente {prediction}")
