from constants import *
from keras import Sequential
from keras.layers import Dense, LSTM, Reshape
from keras.models import load_model


def one_shot_learning(pretrained_model, one_shot_data, one_shot_label):
    # freeze the model layers
    for layer in pretrained_model.layers:
        layer.trainable = False

    # Add new layers on top of the pre-trained model
    new_model = Sequential()
    new_model.add(pretrained_model)
    new_model.add(Dense(64, activation='relu'))
    new_model.add(Dense(len(SENSOR_NAMES), activation='sigmoid'))

    # Compile the new model
    new_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fine-tune the new model
    new_model.fit(one_shot_data, one_shot_label, epochs=2)

    return new_model
