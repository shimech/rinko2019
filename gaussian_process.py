import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
from dataset_generater import DatasetGenerater
from base_functions import BaseFunctions
from visualizer import Visualizer


X_MIN = -3
X_MAX = 3
N_DATA = 100
NOISE_AMP = 0.2
BASE_FUNC = BaseFunctions.func1


def main():
    dg = DatasetGenerater(X_MIN, X_MAX, N_DATA, NOISE_AMP, BASE_FUNC, is_noisy=True)
    X, Y = dg.generate_dateset()
    train_data_index = np.random.randint(0, N_DATA, int(N_DATA * 0.1))
    X_train, Y_train = X[train_data_index], Y[train_data_index]
    X, X_train = X.reshape((-1, 1)), X_train.reshape((-1, 1))
    kernels = {
        "Gaussian Kernel": 0.594 ** 2 * RBF(length_scale=0.279),
        "Product": DotProduct(),
    }
    for name, kernel in kernels.items():
        model_gpr = GaussianProcessRegressor(kernel=kernel)
        reg = model_gpr.fit(X_train, Y_train)
        Y_pred, Y_std = reg.predict(X, return_std=True)
        Visualizer.visualize_gaussian_process(X, Y, Y_pred, Y_std, title=name)


if __name__ == "__main__":
    main()
