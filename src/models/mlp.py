from torch import nn
from typing import List

class MLP(nn.Module):
    def __init__(self, input_size: int=512, output_size: int=10, hidden_layers: List[int]=[256],
                dropout: float=0.0):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
        # Define layer sizes
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        print(f"MLP layers sizes {self.layer_sizes}")

        # Adding layers
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = self.relu(layer(x))
                x = self.dropout(x)
            else:
                x = layer(x)  # No activation or dropout after the last layer
        return x