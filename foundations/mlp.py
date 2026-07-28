import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # h = []
        # h[0] = np.maximum(np.dot(x,weights[0]+biases[0]))
        for i in range (0,(len(weights)-1)):
      
            x = np.dot(x, weights[i]) + biases[i]

            x = np.maximum(0, x)
        z = np.dot(x,weights[-1])+biases[-1]


        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        return np.round(z, 5)
        
