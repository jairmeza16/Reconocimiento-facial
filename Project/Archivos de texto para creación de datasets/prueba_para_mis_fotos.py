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

files_names_me = os.listdir('C:/Users/jairm/OneDrive/Documentos/Repositorios/Reconocimiento_facial/Project/data/train/me')
attributes_me = [1] * 6642
df = pd.DataFrame((zip(files_names_me, attributes_me)))
print(df.shape)
df_me = pd.DataFrame(files_names_me)
print(len(files_names_me))
