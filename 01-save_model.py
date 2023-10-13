import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

cifar10_class_names = {
    0:"Plane",
    1:"Car",
    2:"Bird",
    3:"Cat",
    4:"Deer",
    5:"Dog",
    6:"Frog",
    7:"Horse",
    8:"Boat",
    9:"Truck",
}

cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

 

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train= x_train/255
x_test = x_test /255

 

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

 

model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(32,32,3)))
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dense(250, activation='relu'))
model.add(keras.layers.Dense(10, activation="softmax"))



model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )

 
result = model.fit(x_train, y_train,\
                   batch_size=32,\
                   epochs=100,\
                   validation_data=[x_test, y_test],\
                   shuffle=True)

 



model.save("models/cifar10_model.h5")


