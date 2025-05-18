# config.py

# === Dataset Paths ===
DATASET_PATH = "./input/usps.h5"
TEACHER_IMAGE_PATH = "./digit/digit_{}.jpg"
STUDENT_IMAGE_PATH = "./digit/digit_s_{}.jpg"

# === Image Dimensions ===
IMAGE_SIZE = (16, 16)
NUM_CLASSES = 10
FLAT_IMAGE_SIZE = 256

# === Classes ===
DIGIT_CLASSES = list(range(10))

# === Training Parameters ===
SVD_MAX_BASIS = 12
HOSVD_MAX_BASIS = 12
CNN_EPOCHS = 20
MLP_EPOCHS = 20
RNN_EPOCHS = 20
LSTM_EPOCHS = 20
BATCH_SIZE = 300
VALIDATION_SPLIT = 0.2
SEED = 3

# === Random Forest Parameters ===
RF_ESTIMATORS = 10
RF_CRITERION = 'entropy'
