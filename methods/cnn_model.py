from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from config import IMAGE_SIZE, NUM_CLASSES  # IMAGE_SIZE = (16, 16), NUM_CLASSES = 10

def build_cnn_model():
    model = Sequential()
    model.add(Conv2D(filters=16,
                     kernel_size=(5, 5),
                     padding='same',
                     input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=36,
                     kernel_size=(5, 5),
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_cnn_model(model, x_train, y_train, validation_split=0.2, epochs=20, batch_size=300):
    y_train_oh = to_categorical(y_train, NUM_CLASSES)
    x_train = x_train.reshape((-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)).astype('float32')
    history = model.fit(x_train, y_train_oh, validation_split=validation_split,
                        epochs=epochs, batch_size=batch_size, verbose=2)
    return history

def evaluate_cnn_model(model, x_test, y_test):
    y_test_oh = to_categorical(y_test, NUM_CLASSES)
    x_test = x_test.reshape((-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)).astype('float32')
    return model.evaluate(x_test, y_test_oh, verbose=0)[1]


def predict_cnn_model(model, x):
    x = x.reshape((-1, 16, 16, 1)).astype('float32')
    return model.predict(x).argmax(axis=1)

