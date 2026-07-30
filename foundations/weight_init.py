import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2/(fan_in+fan_out))
        weights=torch.randn(fan_out, fan_in) * std
        return torch.round(weights,decimals=4).tolist()
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        
    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std=math.sqrt(2/fan_in)
        weights=torch.randn(fan_out, fan_in) * std
        return torch.round(weights,decimals=4).tolist()
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        current_fan_in = input_dim
        weights = []
        stds = [] # List to store your matrices
        for i in range(0,num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2/(hidden_dim+current_fan_in))
                w= torch.randn(hidden_dim,current_fan_in)*std
            elif init_type == 'kaiming':
                std = math.sqrt(2/current_fan_in)
                w= torch.randn(hidden_dim,current_fan_in)*std
            elif init_type == 'random':
                w= torch.randn(hidden_dim,current_fan_in)
            weights.append(w)
           
            current_fan_in = hidden_dim
        x=torch.randn(input_dim)
        for w in weights:
            x=w@x
            x = torch.relu(x)
            stds.append(round(x.std().item(),2))
        return stds

        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
      
