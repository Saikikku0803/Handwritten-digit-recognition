# simple_mean.py
import numpy as np
import pandas as pd
from config import IMAGE_SIZE

def norm2(a, b):
    """Compute 2-norm distance between two images (reshaped to 2D)."""
    return np.linalg.norm(a - b, 2)

def compute_mean_images(x_data, y_data):
    """
    Calculate mean image for each digit class.
    Returns a DataFrame of shape (10, 257): 256 pixels + label.
    """
    df = pd.concat([pd.DataFrame(x_data), pd.DataFrame(y_data, columns=['label'])], axis=1)
    mean_table = np.zeros((10, 257))
    for digit in range(10):
        class_data = df[df['label'] == digit].iloc[:, :256]
        mean_table[digit, :-1] = class_data.mean(axis=0)
        mean_table[digit, -1] = digit
    return pd.DataFrame(mean_table, columns=[*range(256), 'label'])

def predict_with_mean(x_test_df, mean_df):
    """
    Predict digits by comparing test images with class mean images using 2-norm.
    Returns predicted DataFrame and accuracy.
    """
    num_samples = x_test_df.shape[0]
    predictions = np.zeros(num_samples, dtype=int)
    residuals = np.zeros(num_samples)
    residual_list = []

    for i in range(num_samples):
        test_img = x_test_df.iloc[i, :256].values.reshape(IMAGE_SIZE)
        dists = []
        for j in range(10):
            mean_img = mean_df.iloc[j, :256].values.reshape(IMAGE_SIZE)
            dist = norm2(test_img, mean_img)
            dists.append(dist)
        predicted_label = np.argmin(dists)
        predictions[i] = predicted_label
        residuals[i] = dists[predicted_label]
        residual_list.append(dists)

    result_df = pd.concat([
        x_test_df.reset_index(drop=True),
        pd.DataFrame(predictions, columns=['prediction']),
        pd.DataFrame(residuals, columns=['residual'])
    ], axis=1)

    accuracy = (result_df['label'] == result_df['prediction']).mean()
    return result_df, accuracy, residual_list
