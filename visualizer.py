import matplotlib.pyplot as plt
from dataset_generater import DatasetGenerater
from base_functions import BaseFunctions


class Visualizer:
    @staticmethod
    def visualize_dataset(X, Y, base_func):
        plt.figure()
        plt.scatter(X, Y, label="Observed Value", marker=".", color="green")
        plt.plot(X, base_func(X), label="True Value", color="blue")
        plt.legend()
        print("データセットを表示しました。")
        plt.show()

    @staticmethod
    def visualize_regression(X, Y, Y_pred, title=None):
        plt.figure()
        plt.scatter(X, Y, label="Observed Value", marker=".", color="green")
        plt.plot(X, Y_pred, label="Predicted Value", color="blue")
        plt.legend()
        if title is not None:
            plt.title(title)
        print("回帰: {} を表示しました。".format(title))
        plt.show()
    
    @staticmethod
    def visualize_gaussian_process(X, Y, Y_pred, Y_std, title=None):
        plt.figure()
        plt.scatter(X, Y, label="Observed Value", marker=".", color="green")
        plt.plot(X, Y_pred, label="Predicted Value", color="blue")
        plt.fill_between(X[:, 0], Y_pred - Y_std, Y_pred + Y_std, color="orange", alpha=0.2)
        plt.legend()
        if title is not None:
            plt.title(title)
        print("回帰: {} を表示しました。".format(title))
        plt.show()


if __name__ == "__main__":
    dg = DatasetGenerater(x_min=-3, x_max=3, n_data=100, noise_amp=0.2, base_func=BaseFunctions.func1, is_noisy=True)
    X, Y = dg.generate_dateset()
    Visualizer.visualize_dataset(X, Y, dg.base_func)
