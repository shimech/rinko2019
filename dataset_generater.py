import numpy as np
from base_functions import BaseFunctions as bf


class DatasetGenerater:
    def __init__(self, x_min, x_max, n_data, noise_amp, base_func, is_noisy=True):
        self.x_min = x_min
        self.x_max = x_max
        self.n_data = n_data
        self.noise_amp = noise_amp
        self.base_func = base_func
        self.is_noisy = is_noisy

    def generate_dateset(self):
        X = np.linspace(start=self.x_min, stop=self.x_max, num=self.n_data)
        Y = self.base_func(X)
        if self.is_noisy:
            noise = self.__generate_noise()
            Y += noise
        return X, Y

    def __generate_noise(self):
        return self.noise_amp * np.random.randn(self.n_data)
