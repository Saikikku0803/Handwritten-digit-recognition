# svd_classification.py

import numpy as np
import pandas as pd
from config import IMAGE_SIZE

def get_svd_bases(x_train, y_train):
    """
    建立每一類數字 (0~9) 的 SVD 基底向量（U 矩陣）。
    回傳一個 list，每個元素是該類別的 U 矩陣。
    """
    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_train = x_train.T  # shape: (256, N)

    svd_bases = []
    for digit in range(10):
        indices = np.where(y_train == digit)[0]
        digit_samples = x_train[:, indices]
        U, _, _ = np.linalg.svd(digit_samples, full_matrices=False)
        svd_bases.append(U)
    return svd_bases

def svd_predict(x_data, y_data, svd_bases, max_basis=12):
    """
    使用每一類別的前 r 個 SVD 基底進行投影，計算投影誤差後分類。
    回傳 result_df、準確率、residual list、mse list。
    """
    x_data = x_data.astype('float32') / 255.
    x_data = x_data.reshape((len(x_data), np.prod(x_data.shape[1:])))
    x_data = x_data.T  # shape: (256, N)

    residual_matrix = []
    mse_list = []

    for r in range(max_basis):
        N = x_data.shape[1]
        predictions = np.zeros(N, dtype=int)
        residuals = np.zeros(N)
        all_residuals = []

        for i in range(N):
            z = x_data[:, i]
            dist_list = []
            for digit in range(10):
                U = svd_bases[digit][:, :r+1]
                proj = U @ (U.T @ z)
                dist = np.linalg.norm(z - proj, 2) / np.linalg.norm(z, 2)
                dist_list.append(dist)
            pred = np.argmin(dist_list)
            predictions[i] = pred
            residuals[i] = dist_list[pred]
            all_residuals.append(dist_list)

        mse = np.mean((y_data - predictions) ** 2)
        residual_matrix.append((predictions, residuals, all_residuals))
        mse_list.append(mse)

    # 取用最後一層結果
    predictions, residuals, all_residuals = residual_matrix[-1]

    result_df = pd.concat([
        pd.DataFrame(x_data.T),
        pd.DataFrame(y_data, columns=['actual']),
        pd.DataFrame(predictions, columns=['prediction']),
        pd.DataFrame(residuals, columns=['residual'])
    ], axis=1)

    accuracy = (result_df['actual'] == result_df['prediction']).mean()

    return result_df, predictions, accuracy, all_residuals, mse_list
