import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        z= np.dot(w,x)+b
        y_pred = 1/(1+np.exp(-z))
        dL_dw = (x*(y_pred - y_true)*(y_pred*(1-y_pred)))
        dL_db = (y_pred - y_true)*(y_pred*(1-y_pred))
        return (np.round(dL_dw,5),np.round(dL_db,5))
        
       
