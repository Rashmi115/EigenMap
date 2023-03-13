# Import necessary libraries
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import json

def build_cnn():
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess the data
    X_train = X_train.reshape((60000, 28, 28, 1))
    X_test = X_test.reshape((10000, 28, 28, 1))
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Create the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    measures = classification_report(y_test, y_pred)
    print(measures)
    
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
    json_measures = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "fscore": fscore.tolist()
    }
    
    return [ model, json_measures ]

# build_cnn()
