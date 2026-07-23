import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        s = np.dot(x,w) + b
        if activation == "sigmoid":
            z=(1)/(1+np.exp(-s))
        else:
            z=max(0.0,s)
        return np.round(z,5)


        
    
