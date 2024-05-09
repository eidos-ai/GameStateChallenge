from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
import numpy as np
from PIL import Image

from models.convnext import ConvNext


class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        
        # Initialize dataset with random images in the correct format and range
        self.data = np.random.randint(0, 256, (100, 3, 224, 224), dtype=np.uint8)
        
        # Example targets
        self.targets = np.random.randint(0, 10, (100,))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        
        # Convert the numpy array to a PIL Image
        image = Image.fromarray(x.transpose(1, 2, 0))
        
        y = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, y
    
class JerseyNumTrainer(pl.LightningModule):
    def __init__(self):       
        super().__init__()
        # First create model
        self.model = ConvNext("convnextv2_nano.fcmae_ft_in1k", 100)
        
        # Then dataloaders, using model's transform
        # TODO: add params here
        # self.train_loader = init torch DataLoader for train dataset
        # self.valid_loader = init torch DataLoader for val dataset
        self.setup_dummy_data()
        
        
    def setup_dummy_data(self):
        self.train_transform = self.model.get_transform(is_training= True)
        self.train_dataset = CustomDataset(transform=self.train_transform)
        
        self.inference_transform = self.model.get_transform(is_training= False)
        self.val_dataset = CustomDataset(transform=self.inference_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=6, shuffle = True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=6,  shuffle = True)
        # TODO: Read batch_size as argument on main script
    
    def _common_step(self, batch, batch_id):
        x_batch, gt_batch = batch
        pred_y_batch = self.model(x_batch)
        loss = self.model.loss(pred_y_batch, gt_batch)
        return pred_y_batch, loss
    
    def training_step(self, batch, batch_idx):
        pred_y_batch, loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):        
        _, gt_batch = batch
        pred_y_batch, loss = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        pred_labels = self.model.pred_to_label(pred_y_batch)
        # TODO: log accuracy pred_labels vs gt_batch
        return loss
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()
    
def train():
    in_trainer = JerseyNumTrainer()
    trainer = pl.Trainer(
        max_epochs=10, #TODO: Read epochs as argument on main script
        log_every_n_steps=10
    )
    # Train the model
    trainer.fit(in_trainer, in_trainer.train_loader, in_trainer.val_loader)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--train', default="../data/jersey_num/train", required=False, help='Path to directory with JerseyDataset train split')
    parser.add_argument('-v', '--val', default="../data/jersey_num/val", required=False, help='Path to directory with JerseyDataset val split')
    parser.add_argument('-te', '--test', default="../data/jersey_num/test", required=False, help='Path to directory with JerseyDataset test split')
    args = vars(parser.parse_args())
    
    print("Ho")
    train()