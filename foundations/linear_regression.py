import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        s=np.dot(X,weights)
        return np.round(s,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        s=(1/len(model_prediction))*(np.sum((model_prediction - ground_truth)**2))
        return np.round(s,5)
        
