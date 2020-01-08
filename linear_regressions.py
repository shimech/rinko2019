import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
    X = X.reshape((-1, 1))
    features = {
        "Linear": X,
        "2 degree": PolynomialFeatures(degree=2).fit_transform(X),
        "5 degree": PolynomialFeatures(degree=5).fit_transform(X),
        "10 degree": PolynomialFeatures(degree=10).fit_transform(X),
        "100 degree": PolynomialFeatures(degree=100).fit_transform(X),
    }
    model_lr = LinearRegression()
    for name, feature in features.items():
        reg = model_lr.fit(feature, Y)
        Visualizer.visualize_regression(X, Y, reg.predict(feature), title=name)


if __name__ == "__main__":
    main()
