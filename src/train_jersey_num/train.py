import os
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
import numpy as np
from PIL import Image

from models.convnext import ConvNext
from dataloader import JerseyNumberDataset
    
class JerseyNumTrainer(pl.LightningModule):
    def __init__(self, train_datasetpath, valid_datasetpath, test_datasetpath, t_batch_size, v_batch_size):       
        super().__init__()
        # First create model
        self.model = ConvNext("convnextv2_nano.fcmae_ft_in1k", 100)
        
        # Then dataloaders, using model's transform
        self.train_transform = self.model.get_transform(is_training= True)
        self.train_dataset = JerseyNumberDataset(dataset_folder=train_datasetpath, transform=self.train_transform, include_number_not_seen=True)
        
        self.inference_transform = self.model.get_transform(is_training= False)
        self.val_dataset = JerseyNumberDataset(dataset_folder=test_datasetpath, transform=self.inference_transform, include_number_not_seen=True)

        self.train_loader = DataLoader(self.train_dataset, batch_size=t_batch_size, shuffle = False)
        self.val_loader = DataLoader(self.val_dataset, batch_size=v_batch_size,  shuffle = False)
    
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
        
        accuracy = torch.sum(pred_labels == gt_batch) / float(gt_batch.size(0))
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()
    
def train(train_datasetpath, valid_datasetpath, test_datasetpath, epochs, t_batch_size, v_batch_size):
    in_trainer = JerseyNumTrainer(train_datasetpath, test_datasetpath, test_datasetpath, t_batch_size, v_batch_size)
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10
    )
    # Train the Trainer
    trainer.fit(in_trainer, in_trainer.train_loader, in_trainer.val_loader)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--train', default="train", required=False, help='Path to directory with JerseyDataset train split')
    parser.add_argument('-v', '--val', default="valid", required=False, help='Path to directory with JerseyDataset val split')
    parser.add_argument('-te', '--test', default="test", required=False, help='Path to directory with JerseyDataset test split')
    parser.add_argument('-ep', '--epochs', type=int, default=10, required=False, help='Number of epochs for training')
    parser.add_argument('-tb', '--train_batch', type=int, default=6, required=False, help='Batch size for training data')
    parser.add_argument('-vb', '--val_batch', type=int, default=6, required=False, help='Batch size for validation data')
    args = vars(parser.parse_args())

    if not os.path.exists(args['train']):
        print(f"Error: Path '{args['train']}' does not exist.")
        exit(1)
    if not os.path.exists(args['val']):
        print(f"Error: Path '{args['val']}' does not exist.")
        exit(1)
    if not os.path.exists(args['test']):
        print(f"Error: Path '{args['test']}' does not exist.")
        exit(1)

    # print("Hi")
    train(args['train'], args['val'], args['test'], args['epochs'], args['train_batch'], args['val_batch'])