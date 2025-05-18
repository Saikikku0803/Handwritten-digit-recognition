import h5py
import numpy as np
from PIL import Image, ImageOps
import os

def load_usps_dataset(path):
    with h5py.File(path, 'r') as hf:
        x_train = hf['train/data'][:]
        y_train = hf['train/target'][:]
        x_test = hf['test/data'][:]
        y_test = hf['test/target'][:]
    return x_train, y_train, x_test, y_test



def preprocess_image(path, size=(16, 16)):
    img = Image.open(path).convert("L")  # 灰階
    img = ImageOps.invert(img)           # 反相
    img = img.resize(size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = (img_array > 128).astype(np.uint8) * 255  # 二值化為 0 或 255
    return img_array.flatten()

def load_digit_images(folder_path, prefix="digit_", size=(16, 16)):
    images = []
    for i in range(10):
        path = os.path.join(folder_path, f"{prefix}{i}.jpg")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        images.append(preprocess_image(path, size))
    return np.array(images)

def load_student_images(folder_path, prefix="digit_s_", size=(16, 16)):
    images = []
    for i in range(10):
        path = os.path.join(folder_path, f"{prefix}{i}.jpg")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        images.append(preprocess_image(path, size))
    return np.array(images)