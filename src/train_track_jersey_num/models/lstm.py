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

class SampleLSTM(pl.LightningModule):
    def __init__(self,
        learning_rate,
        hidden_size = 128,
        num_layers = 2
    ):
        super().__init__()
        self.learning_rate = learning_rate
        input_size = 200
        num_classes = 100
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, (hidden, _) = self.lstm(x)
        # Take the output of the last sequence step
        out = self.fc(hidden[-1])        
        out = F.softmax(out, dim=1)
        return out
        
    
    def loss(self, pred, target):
        loss = F.nll_loss(torch.log(pred), target) 
        return loss
    
    def pred_to_label(self, pred):
        # logging.debug(f"pred shape {pred.shape}")
        _, pred_class = torch.max(pred, 1)
        # logging.debug(f"pred_class {pred_class}") 
        return pred_class
                        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
        
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))

