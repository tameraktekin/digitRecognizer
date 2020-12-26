import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.regularizers import l1
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def read_data(data_dir, is_train):
    data = pd.read_csv(data_dir)
    if is_train:
        data = data.sample(frac=1).reset_index(drop=True)
        labels = data["label"]
        labels = to_categorical(labels)
        data = data.drop("label", axis=1)
        data /= 255

        data_train, labels_train, data_val, labels_val = train_test_split(data, labels, test_size=0.2, random_state=42)
        return data_train, labels_train, data_val, labels_val
    else:
        data /= 255
        return data


def build_model(layer_size, regularizer, drop_size, num_of_classes, lr):
    model = Sequential()

    model.add(Dense(layer_size, activity_regularizer=regularizer))
    model.add(BatchNormalization)
    model.add(Activation('relu'))
    model.add(Dropout(drop_size))

    model.add(Dense(layer_size, activity_regularizer=regularizer))
    model.add(BatchNormalization)
    model.add(Activation('relu'))
    model.add(Dropout(drop_size))

    model.add(Dense(num_of_classes))
    model.add(Activation("softmax"))

    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


train_data_dir = "./digit-recognizer/train.csv"
test_data_dir = "./digit-recognizer/test.csv"

train_data, train_labels, val_data, val_labels = read_data(train_data_dir, True)
test_data = read_data(test_data_dir, False)

