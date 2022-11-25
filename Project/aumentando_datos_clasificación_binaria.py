import tensorflow as tf
#import tensorflow_datasets as tfds
import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
#from sklearn.model_selection import train_test_split
import pathlib
import datetime
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import IPython.display as display
from PIL import Image
from turtle import pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'






#En las siguientes líneas estoy creando las carpetas para albergar mis imágenes y aumentarlas. Como ya corrí el código una vez, comentaré esta parte para proseguir.

# Home directory
home_path = r'C:/Users/jairm/OneDrive/Documentos/Redes neuronales/Reconocimiento_facial/img_align_celeba/fotos_propias'

'''# Create train and validation directories
train_path = os.path.join(home_path,'train')
os.mkdir(train_path)
val_path = os.path.join(home_path,'valid')
os.mkdir(val_path)


# Create sub-directories
me_train_path = os.path.join(home_path + r'/train','me')
os.mkdir(me_train_path)

not_me_train_path = os.path.join(home_path + r'/train','not_me')
os.mkdir(not_me_train_path)

me_val_path = os.path.join(home_path + r'/valid','me')
os.mkdir(me_val_path)

not_me_val_path = os.path.join(home_path + r'/valid','not_me')
os.mkdir(not_me_val_path)'''

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory to store augmentation examples")
ap.add_argument("-t", "--total", type=int, default=100,
	help="# of training samples to generate")
args = vars(ap.parse_args())

# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension
print("[INFO] loading example image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# construct the image generator for data augmentation then
# initialize the total number of images generated thus far
aug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
total = 0

# construct the actual Python generator
print("[INFO] generating images...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
	save_prefix="image", save_format="jpg")
# loop over examples from our image data augmentation generator
for image in imageGen:
	# increment our counter
	total += 1
	# if we have reached the specified number of examples, break
	# from the loop
	if total == args["total"]:
		break





