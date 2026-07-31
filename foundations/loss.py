import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        s=-np.mean(y_true*np.log(y_pred + (1e-7))+(1-y_true)*(np.log((1-y_pred)+1e-7)))
        return np.round(s,4)
    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        s = -(1/len(y_true))*np.sum(((y_true)*np.log(y_pred+(1e-7))))
        return round(s, 4)
       
