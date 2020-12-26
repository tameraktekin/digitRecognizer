import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.regularizers import l1
import numpy as np
import pandas as pd


def read_data(data_dir, is_train):
    data = pd.read_csv(data_dir)
    if is_train:
        data    = data.sample(frac=1).reset_index(drop=True)
        labels  = data["label"]
        labels  = to_categorical(labels)
        data    = data.drop("label", axis=1)
        data    /= 255
        return data, labels
    else:
        data /= 255
        return data


def build_model(layer_size, input_shape, regularizer, drop_size, num_of_classes, lr):
    model = Sequential()

    model.add(Dense(layer_size, activity_regularizer=regularizer, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_size))

    model.add(Dense(layer_size, activity_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_size))

    model.add(Dense(num_of_classes))
    model.add(Activation("softmax"))

    optimizer = optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


train_data_dir  = "./digit-recognizer/train.csv"
test_data_dir   = "./digit-recognizer/test.csv"

train_data, train_labels = read_data(train_data_dir, True)
test_data = read_data(test_data_dir, False)

layer_size      = 1024
input_shape     = train_data.shape
regularizer     = l1(0.001)
drop_size       = 0.2
num_of_classes  = 10
lr              = 1e-5

model = build_model(layer_size, input_shape, regularizer, drop_size, num_of_classes, lr)
model.summary()
