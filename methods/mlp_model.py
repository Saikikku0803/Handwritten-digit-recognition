from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from config import IMAGE_SIZE, NUM_CLASSES

def build_mlp_model(input_dim=IMAGE_SIZE[0] * IMAGE_SIZE[1]):
    model = Sequential()
    model.add(Dense(units=256, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=NUM_CLASSES, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_mlp_model(x_train, y_train, epochs=20, batch_size=300):
    y_train_onehot = to_categorical(y_train, NUM_CLASSES)
    model = build_mlp_model()
    history = model.fit(
        x=x_train,
        y=y_train_onehot,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    return model, history

def evaluate_mlp_model(model, x_test, y_test):
    y_test_onehot = to_categorical(y_test, NUM_CLASSES)
    loss, accuracy = model.evaluate(x_test, y_test_onehot, verbose=0)
    return accuracy

def predict_images(model, x):
    predictions = model.predict(x)
    return predictions.argmax(axis=1)