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

    def __sigmoid(self, z: Union[np.array, float]) -> Union[np.array, float]:
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
            y_pred = self.__sigmoid(-z)
            loss = self.__log_loss(y_pred, y)
            grads = (y_pred-y) @ X /len(X)
            self.weights -= self.learning_rate * grads
            if verbose:
                if i % verbose == 0:
                    z = np.dot(self.weights, X.T)
                    y_pred = self.__sigmoid(-z)
                    loss = np.mean(y*np.log(y_pred + 1e-15) + (1-y)*np.log(1-y_pred + 1e-15))
                    print(f"{i} | {loss}")

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        X = add_fictive(X.values)
        z = np.dot(self.weights, X.T)
        y_pred = self.__sigmoid(-z)
        return y_pred

    def predict(self, X:pd.DataFrame) -> np.array:
        return (self.predict_proba(X) > 0.5).astype(int)

    def calc_TP(self, y: np.array, y_pred: np.array) -> float:
        return ((y_pred == 1) & (y == 1)).sum()

    def calc_FP(self, y: np.array, y_pred: np.array) -> float:
        return ((y_pred == 1) & (y == 0)).sum()

    def calc_TN(self, y: np.array, y_pred: np.array) -> float:
        return ((y_pred == 0) & (y == 0)).sum()

    def calc_FN(self, y: np.array, y_pred: np.array) -> float:
        return ((y_pred == 0) & (y == 1)).sum()

    def calc_accuracy(self, y: np.array, y_pred: np.array) -> float:
        return (y == y_pred).sum() / len(y)

    def calc_precision(self, y: np.array, y_pred: np.array) -> float:
        TP = self.calc_TP(y, y_pred)
        FP = self.calc_FP(y, y_pred)
        return TP / (TP+FP)

    def calc_recall(self, y: np.array, y_pred: np.array) -> float:
        TP = self.calc_TP(y, y_pred)
        FN = self.calc_FN(y, y_pred)
        return TP / (TP+FN)

    def calc_F1(self, y: np.array, y_pred: np.array) -> float:
        precision = self.calc_precision(y, y_pred)
        recall = self.calc_recall(y, y_pred)
        return 2 * (precision * recall) / (precision + recall)

    def calc_roc_auc(self, y: np.array, y_pred_proba: np.array) -> float:
        positive_samples: int = (y == 1).sum()
        negative_samples: int = (y == 0).sum()
        asc_score_indices = np.argsort(y_pred_proba)
        y = np.flip(y[asc_score_indices])
        scores = np.flip(y_pred_proba[asc_score_indices])
        roc_auc_sum: float = 0
        for i in range(len(y)):
            if y[i] == 0:
                for j in range(i):
                    
                    if scores[j] > scores[i]:
                        print(f"{scores[j]}>{scores[i]}")
                        roc_auc_sum += 1
                    elif scores[j] == scores[i]:
                        print(f"{scores[j]}=={scores[i]}")
                        roc_auc_sum += 0.5
        return (1 / (positive_samples * negative_samples)) * roc_auc_sum
