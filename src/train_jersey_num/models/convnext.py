import pytorch_lightning as pl
import torch
import timm
from typing import List
from pathlib import Path
from torch.nn import functional as F
import logging

#For logging output
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)


class ConvNext(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int, learning_rate=1e-3):
        super().__init__()
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Create the base model with pretrained weights but without the classifier layer
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=num_classes
        )
        
        # Get model specific transforms (normalization, resize)
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.inference_transform = timm.data.create_transform(**self.data_config, is_training=False)
        self.train_transform = timm.data.create_transform(**self.data_config, is_training=True)
    
    def get_transform(self, is_training: bool):
        return timm.data.create_transform(**self.data_config, is_training=is_training)

    def forward(self, x):
        logging.debug(f"Model Input x {x.shape}")
        out = self.model(x)
        logging.debug(f"Model Output x {x.shape}")
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
    
    def preprocess_imgs(self, imgs_paths: List[Path], device: str="cuda"):
        images = [Image.open(str(p)).convert("RGB") for p in imgs_paths]
        preprocessed_images = [self.inference_transform(image).to(device) for image in images]
        preprocessed_images = torch.stack(preprocessed_images, dim=0)
        return preprocessed_images
    
    def img_features(self, imgs_paths: List[Path], device: str="cuda"):
        with torch.no_grad():
            return self(self.preprocess_imgs(imgs_paths, device))