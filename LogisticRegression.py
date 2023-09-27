import numpy as np
import pandas as pd
from typing import NoReturn, Union

def add_fictive(X: np.array) -> np.array:
    return np.insert(X, 0, np.ones(len(X)), axis=1)

class MyLogReg:
    def __init__(self, n_iter: int = 10, learning_rate: float = 0.1) -> NoReturn:
        self.n_iter: int = n_iter
        self.learning_rate: float = learning_rate
        self.weights: np.array = np.array([])

    def __str__(self) -> str:
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def __call__(self) -> str:
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def __sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(z))

    def __log_loss(self, y_pred: np.array, y: np.array):
        return -np.mean(y*np.log(y_pred + 1e-15) + (1-y)*np.log(1-y_pred + 1e-15))

    def get_coef(self) -> np.array:
        return self.weights[1:]

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[bool, int]) -> NoReturn :
        X = add_fictive(X.values)
        y = y.values
        self.weights = np.ones(len(X[0]))
        if verbose:
            z = np.dot(self.weights, X.T)
            y_pred = self.__sigmoid(-z)
            loss = self.__log_loss(y_pred, y)
            print(f"start | {loss}")
        for i in range(1, self.n_iter+1):
            z = np.dot(self.weights, X.T)
            y_pred = y_pred = self.__sigmoid(-z)
            loss = self.__log_loss(y_pred, y)
            grads = (y_pred-y) @ X /len(X)
            self.weights -= self.learning_rate * grads
            if verbose:
                if i % verbose == 0:
                    z = np.dot(self.weights, X.T)
                    y_pred = y_pred = self.__sigmoid(-z)
                    loss = np.mean(y*np.log(y_pred + 1e-15) + (1-y)*np.log(1-y_pred + 1e-15))
                    print(f"{i} | {loss}")
