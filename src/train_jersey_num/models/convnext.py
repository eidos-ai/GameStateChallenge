import pytorch_lightning as pl
import torch
import timm
from typing import List
from pathlib import Path
from torch.nn import functional as F
import logging
from PIL import ImageOps
import torchvision.transforms as transforms

#For logging output
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging.getLogger().setLevel(logging.INFO)

def letterbox_image(image, output_size):
    # Calculate padding size to maintain aspect ratio
    width, height = image.size
    max_dim = max(width, height)
    padding_left = (max_dim - width) // 2
    padding_right = max_dim - width - padding_left
    padding_top = (max_dim - height) // 2
    padding_bottom = max_dim - height - padding_top

    # Add padding
    image = ImageOps.expand(image, (padding_left, padding_top, padding_right, padding_bottom))

    # Resize and crop
    image = transforms.Resize(output_size, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop(output_size)(image)
    return image

class ConvNext(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int, learning_rate):
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
        if not is_training:            
            transform = transforms.Compose([
                transforms.Lambda(lambda img: letterbox_image(img, 256)),
                # transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
                # transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # transform = transforms.Compose([
            #     # transforms.Lambda(lambda img: letterbox_image(img, 256)),
            #     transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.333), interpolation=transforms.InterpolationMode.BICUBIC),
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4]),
        
            #     # transforms.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0), ratio=(0.2, 1.333), interpolation=transforms.InterpolationMode.BICUBIC),
            #     # transforms.ColorJitter(brightness=0.5, contrast=1, saturation=1, hue=0.3),
            #     # transforms.RandomRotation(degrees=30),
            #     # transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
            #     # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # ])
            transform = self.train_transform
        return transform
        

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
        
    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
