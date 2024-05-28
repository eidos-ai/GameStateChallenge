import pytorch_lightning as pl
import torch
import math
import timm
from typing import List
from pathlib import Path
from torch.nn import functional as F
import logging
import torch.nn as nn

#For logging output
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, device: torch.device = "cpu"):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.device = device
        self.positional_encoding = self.generate_positional_encoding()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional_encoding
        return x

    def generate_positional_encoding(self) -> torch.Tensor:
        positional_encoding = torch.zeros(self.max_length, self.d_model)
        position = torch.arange(0, self.max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).to(self.device)
        return positional_encoding

class SampleFusion(pl.LightningModule):
    def __init__(self,
        learning_rate,
        num_layers: int,
        d_model: int,
        num_heads: int,
        max_length: int,
        d_ff: int = 1024,
        input_dim: int = 1,
        output_dim: int = 100,
        partial_model: bool = False
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.partial_model = partial_model
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length, "cuda")
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff),
            num_layers,
        )
        self.fc = nn.Linear(d_model, output_dim)
        self.fc_partial = nn.Linear(max_length, 100)
        
    def forward(self, x):
        # print(f"Input: {x.shape}")
        x = self.embedding(x)
        # print(f"Output embedding: {x.shape}")
        # x = self.pos_encoding(x) 
        # print(f"Output positional encoding: {x.shape}")
        x = self.encoder(x)
        # print(f"Output transformer encoder: {x.shape}")
        x = x[:,-1,:]
        x = self.fc(x)
        # print(f"Output fully connected: {x.shape}")
        # x = F.softmax(x, dim=1
        # print(f"outfor forwar {x.shape}")
        x = F.softmax(x, dim=1)
        return x
    
    def loss(self, pred, target):
        # logging.debug(f"loss pred {pred.shape}")
        # logging.debug(f"loss target {target.shape}")
        # loss = F.cross_entropy(pred, target)
        loss = F.nll_loss(torch.log(pred), target) 
        # print(loss)
        return loss
    
    def pred_to_label(self, pred):
        logging.debug(f"pred shape {pred.shape}")
        _, pred_class = torch.max(pred, 1)
        logging.debug(f"pred_class {pred_class}") 
        return pred_class
                        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
        
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
