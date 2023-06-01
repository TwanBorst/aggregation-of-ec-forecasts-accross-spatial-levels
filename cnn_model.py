from keras import Sequential
from keras.layers import MaxPooling1D, Flatten, Dense, Conv1D
from sklearn.model_selection import train_test_split


def train_model(training_data, training_labels):
    X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size=0.2, random_state=42)

    # Create the CNN model
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(training_labels.shape[1], activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    return model
