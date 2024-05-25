import pytorch_lightning as pl
import torch
import timm
from typing import List
from pathlib import Path
from torch.nn import functional as F
import logging
import torch.nn as nn

#For logging output
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


class CNNFusion(pl.LightningModule):
    def __init__(self, learning_rate, input_channels=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.conv1 = nn.Conv1d(input_channels, 20, 3, stride=1, padding=1, bias=True)
        self.conv2_1 = nn.Conv1d(20, 50, 3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv1d(50, 50, 3, stride=1, padding=1, bias=True) # input size (50, 101)
        # elementwise sum Conv2 1+Conv2 2
        self.conv3_1 = nn.Conv1d(50, 70, 3, stride=1, padding=1, bias=True) # input size (70, 101)
        self.conv3_2 = nn.Conv1d(70, 70, 3, stride=1, padding=1, bias=True) # input size (70, 101)
        # elementwise sum Conv3 1+Conv3 2
        self.conv4 = nn.Conv1d(70, 70, 2, stride=1, padding=1, bias=True) # input size (102, 70)
        self.max_pool = nn.MaxPool1d(2, stride=2) # size (102, 50)
        self.fc1 = nn.Linear(70*50, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 101)
        
    def forward(self, x):
        print(x.device)
        # logging.info(f"Model Input x {x.shape}")
        # print(f"input size : {x.size()}")
        # x size (1, 101)
        conv1_out = self.conv1(x) # size (20,101)
        # print(f"conv1_out size : {conv1_out.size()}")

        conv2_1_out = self.conv2_1(conv1_out) # size (50, 101)
        # print(f"conv2_1_out size : {conv2_1_out.size()}")
        conv2_2_out = self.conv2_2(conv2_1_out) # size (50, 101)
        # print(f"conv2_2_out size : {conv2_2_out.size()}")
        elem_sum = conv2_1_out + conv2_2_out
        # print(f"elm sum size {elem_sum.size()}") #  size (50, 101)

        conv3_1_out = self.conv3_1(elem_sum)
        # print(f"conv3_1_out size : {conv3_1_out.size()}") # size (70, 101)
        conv3_2_out = self.conv3_2(conv3_1_out)
        # print(f"conv3_2_out size : {conv3_2_out.size()}") # size (70, 101)
        elem_sum_3 = conv3_1_out + conv3_2_out
        # print(f"elm sum 3 size {elem_sum_3.size()}")

        conv4_out = self.conv4(elem_sum_3)
        # print(f"conv4 size {conv4_out.size()}") # size (70, 102)
        max_pool_out = self.max_pool(conv4_out)
        # print(f"max pool size {max_pool_out.size()}") # size (70, 51)

        in_fc1 = torch.flatten(max_pool_out, 1)
        # print(f"input fc1 size {in_fc1.size()}")
        fc1_out = self.fc1(in_fc1)
        fc1_out = F.relu(fc1_out)
        # print(f"input fc2 size {fc1_out.size()}")
        fc2_out = self.fc2(fc1_out)
        fc2_out = F.relu(fc2_out)
        # print(f"fc2 size {fc2_out.size()}")
        out = self.fc3(fc2_out)
        return out
    
    def loss(self, pred, target):
        loss = F.cross_entropy(pred, target)
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
