# cnn_model.py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical

def build_cnn_model(input_shape=(16, 16, 1), num_classes=10):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(36, kernel_size=(5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_cnn_model(model, x_train, y_train, validation_split=0.2, epochs=20, batch_size=300):
    y_train_onehot = to_categorical(y_train)
    return model.fit(x_train, y_train_onehot, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=2)


def evaluate_cnn_model(model, x_test, y_test):
    y_test_onehot = to_categorical(y_test)
    return model.evaluate(x_test, y_test_onehot, verbose=0)[1]


def predict_cnn_model(model, x):
    return model.predict(x).argmax(axis=1)


# rnn_model.py
from keras.models import Sequential
from keras.layers import SimpleRNN, Dropout, Dense
from keras.utils import to_categorical

def build_rnn_model(input_shape=(16, 16), num_classes=10):
    model = Sequential()
    model.add(SimpleRNN(256, input_shape=input_shape, unroll=True))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_rnn_model(model, x_train, y_train, validation_split=0.2, epochs=20, batch_size=300):
    y_train_onehot = to_categorical(y_train)
    return model.fit(x_train, y_train_onehot, validation_split=validation_split, epochs=epochs, batch_size=batch_size, verbose=2)


def evaluate_rnn_model(model, x_test, y_test):
    y_test_onehot = to_categorical(y_test)
    return model.evaluate(x_test, y_test_onehot, verbose=0)[1]


def predict_rnn_model(model, x):
    x = x.reshape((-1, 16, 16, 1)).astype('float32')
    return model.predict(x).argmax(axis=1)

