from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.utils import to_categorical
from utils.data_loader import load_usps_dataset
from config import IMAGE_SIZE
import numpy as np

def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(input_shape=input_shape, units=256, unroll=False))
    model.add(Dropout(0.1))
    model.add(Dense(units=num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_evaluate_lstm(x_train, y_train, x_test, y_test):
    from keras.models import Sequential
    from keras.layers import LSTM, Dropout, Dense
    from keras.utils import to_categorical

    model = Sequential()
    model.add(LSTM(256, input_shape=(16, 16)))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    y_train_oh = to_categorical(y_train)
    model.fit(x_train.reshape(-1, 16, 16), y_train_oh, validation_split=0.2,
              epochs=20, batch_size=300, verbose=2)

    acc = model.evaluate(x_test.reshape(-1, 16, 16), to_categorical(y_test), verbose=0)[1]
    y_pred = model.predict(x_test.reshape(-1, 16, 16)).argmax(axis=1)

    return acc, y_pred

if __name__ == "__main__":
    train_and_evaluate_lstm()
