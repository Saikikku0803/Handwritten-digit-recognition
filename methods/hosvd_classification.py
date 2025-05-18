import numpy as np
import torch as tc
from tensorly.decomposition import tucker
from tensorly import tensor
from config import IMAGE_SIZE


def get_s_and_u_list(x_data, y_data, ranks=(16, 16, 16)):
    x_data = x_data.astype('float32') / 255.
    image_shape = IMAGE_SIZE
    x_list = []

    for k in range(10):
        indices = np.where(y_data == k)[0]
        x_list.append(x_data[indices])

    S_list = []
    U_list = []

    for k in range(10):
        tensor_k = np.zeros((image_shape[0], image_shape[1], x_list[k].shape[0]))
        for i in range(x_list[k].shape[0]):
            tensor_k[:, :, i] = x_list[k][i].reshape(image_shape)

        core, factors = tucker(tensor(tensor_k), rank=ranks)
        S_list.append(core)
        U_list.append(factors)  # [U1, U2, U3]

    return S_list, U_list


def hosvd_algorithm(x_data, y_data, S_list, U_list, basis):
    x_data = x_data.astype('float32') / 255.
    x_data = x_data.reshape((len(x_data), np.prod(x_data.shape[1:])))

    result_mse = []
    result_residual = []

    for r in range(basis):
        n_data = x_data.shape[0]
        y_predict = np.ones(n_data)
        result_single = np.ones(n_data)

        for i in range(n_data):
            Z = x_data[i].reshape(IMAGE_SIZE)
            residual_list = np.ones(10)

            for d in range(10):
                U1, U2, U3 = U_list[d]
                core = S_list[d]

                # 只取前 r+1 個 basis（限制在第三維度）
                Sigma = np.zeros(IMAGE_SIZE)
                for j in range(r + 1):
                    Aj = U1 @ core[:, :, j] @ U2.T
                    Cj = np.tensordot(Z, Aj) / np.tensordot(Aj, Aj)
                    Sigma += Cj * Aj

                G = np.linalg.norm(Z - Sigma, "fro")
                residual_list[d] = G

            y_predict[i] = np.argmin(residual_list)

            if r == basis - 1:
                result_residual.append(residual_list)
                result_single[i] = np.min(residual_list)

        result_mse.append(np.mean((y_data - y_predict) ** 2))

    return y_predict, result_residual, result_mse
