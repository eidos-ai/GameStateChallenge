import os
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import NeptuneLogger
import numpy as np
from PIL import Image

from models.convnext import ConvNext
from dataloader import JerseyNumberDataset
from types import SimpleNamespace

class JerseyNumTrainer(pl.LightningModule):
    def __init__(self, train_datasetpath, valid_datasetpath, test_datasetpath, model_name, batch_size = 32, learning_rate = 1e-4):       
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        
        # First create model
        self.model = ConvNext(model_name, 100, learning_rate)
        
        # Then dataloaders, using model's transform
        self.train_transform = self.model.get_transform(is_training= True)
        self.train_dataset = JerseyNumberDataset(dataset_folder=train_datasetpath, transform=self.train_transform, include_number_not_seen=True)
        
        self.inference_transform = self.model.get_transform(is_training= False)
        self.val_dataset = JerseyNumberDataset(dataset_folder=test_datasetpath, transform=self.inference_transform, include_number_not_seen=True)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,  shuffle = True)
    
    def on_train_start(self):
        self.logger.experiment["training/hyperparams/model_class"] = str(self.model.__class__)
        
        self.logger.experiment["dataset/train_length"] = len(self.train_dataset)
        self.logger.experiment["dataset/val_length"] = len(self.val_dataset)
        
    def _common_step(self, batch, batch_id):
        x_batch, gt_batch = batch
        pred_y_batch = self.model(x_batch)
        loss = self.model.loss(pred_y_batch, gt_batch)
        # print("pred", pred_y_batch)
        # print("gt_batch", gt_batch)
        return pred_y_batch, loss
    
    def training_step(self, batch, batch_idx):
        _, gt_batch = batch
        pred_y_batch, loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        pred_labels = self.model.pred_to_label(pred_y_batch)
        
        accuracy = torch.sum(pred_labels == gt_batch) / float(gt_batch.size(0))
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
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
    
def train(train_datasetpath, valid_datasetpath, test_datasetpath, epochs, batch_size, model_name = "convnextv2_nano.fcmae_ft_in1k"):
    
    in_trainer = JerseyNumTrainer(train_datasetpath, test_datasetpath, test_datasetpath, 
                                  batch_size=batch_size, model_name=model_name)
    
    
    neptune_logger = NeptuneLogger(  # replace with your own
        project="eidos/GameStateChallenge",  # format "workspace-name/project-name"
        tags=["TrainJerseyNumber"],  # optional
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10,
        logger = neptune_logger,
        enable_checkpointing=False
    )
    
    # Train the Trainer
    trainer.fit(
        in_trainer, in_trainer.train_loader, in_trainer.val_loader)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--train', default="../../data/jersey_num/train", required=False, help='Path to directory with JerseyDataset train split')
    parser.add_argument('-v', '--val', default="../../data/jersey_num/valid", required=False, help='Path to directory with JerseyDataset val split')
    parser.add_argument('-te', '--test', default="../../data/jersey_num/test", required=False, help='Path to directory with JerseyDataset test split')
    parser.add_argument('-ep', '--epochs', type=int, default=10, required=False, help='Number of epochs for training')
    parser.add_argument('-tb', '--batch_size', type=int, default=32, required=False, help='Batch size')
    parser.add_argument('-m', '--model_name', type=str, default="convnextv2_nano.fcmae_ft_in1k", required=False, help='ConNext Model to use')
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
    train(args['train'], args['val'], args['test'], 
          epochs = args['epochs'], 
          batch_size = args['batch_size'],
          model_name = args['model_name'])