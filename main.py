from keras.models import Sequential
from keras.layers import Conv2D, Dense, BatchNormalization, Dropout, Activation, MaxPooling2D, Flatten
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.regularizers import l1
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
import time


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


def read_data_as_image(data_dir, is_train):
    data = pd.read_csv(data_dir)
    if is_train:
        data = data.sample(frac=1).reset_index(drop=True)
        labels = data["label"]
        labels = to_categorical(labels)
        data = data.drop("label", axis=1)
        data /= 255
        image_data = data.values
        image_data = image_data.reshape((-1, 28, 28, 1))
        return image_data, labels
    else:
        data /= 255
        image_data = data.values
        image_data = image_data.reshape((-1, 28, 28, 1))
        return image_data


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


def build_cnn_model(layer_size, input_shape, regularizer, drop_size, num_of_classes, lr):
    model = Sequential()

    model.add(Conv2D(layer_size, (3, 3), use_bias=False, input_shape=input_shape,
                     activity_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_size))

    model.add(Conv2D(layer_size, (3, 3), use_bias=False, activity_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_size))

    model.add(Conv2D(layer_size, (3, 3), use_bias=False, activity_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_size))

    model.add(Flatten())
    model.add(Dense(units=int(layer_size/8), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop_size))

    model.add(Dense(units=num_of_classes, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    optimizer = optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


train_data_dir  = "./digit-recognizer/train.csv"
test_data_dir   = "./digit-recognizer/test.csv"

train_data, train_labels = read_data_as_image(train_data_dir, True)
test_data = read_data_as_image(test_data_dir, False)

layer_size      = 1024
# input_shape     = [train_data.shape[1]]
input_shape     = (28, 28, 1)
regularizer     = l1(0.001)
drop_size       = 0.2
num_of_classes  = 10
lr              = 1e-5

model = build_cnn_model(layer_size, input_shape, regularizer, drop_size, num_of_classes, lr)
model.summary()

nb_epochs   = 100
val_split   = 0.2
batch_size  = 32

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-7)

model.fit(train_data, train_labels, epochs=nb_epochs, validation_split=val_split,
          batch_size=batch_size, verbose=1, callbacks=[reduce_lr])

model.save("model" + str(time.time()) + ".h5")

pred    = model.predict(test_data)
preds   = pred.argmax(axis=-1)

results = pd.DataFrame(preds, columns=["Label"])
results.index += 1
results.to_csv("submission" + str(time.time()) + ".csv", index_label="ImageId")