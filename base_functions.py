import numpy as np


class BaseFunctions:
    @staticmethod
    def func1(x):
        return np.sin(np.pi * x) / (np.pi * x) + 0.1 * x
