import os
from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl
import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from torch.utils.data import ConcatDataset
from lightning.pytorch.loggers import NeptuneLogger
import numpy as np
from PIL import Image

from detections_dataset import TrackDetectionsDataset
from models.cnnfusion import CNNFusion
from models.transfusion import TransFusion

@dataclass
class DatasetConfiguration:
    soccernet_train_path: str
    soccernet_test_path: str
    soccernet_valid_path: str
    saved_detections_valid_path: str
    saved_detections_test_path: str
    train_datasets: List[str]
    valid_datasets: List[str]

    def asdict(self):
        return {
            'soccernet_train_path': self.soccernet_train_path,
            'soccernet_test_path': self.soccernet_test_path,
            'soccernet_valid_path': self.soccernet_valid_path,
            'saved_detections_valid': self.saved_detections_valid_path,
            'saved_detections_test': self.saved_detections_test_path,
            'train_datasets': ','.join(self.train_datasets),
            'valid_datasets': ','.join(self.valid_datasets)
        }


class DatasetRegistry():
    def __init__(self, conf: DatasetConfiguration):
        self.conf = conf
        
        # self.soccernet_train_dataset = TrackDetectionsDataset(conf.soccernet_train_path)
        self.soccernet_test_dataset = TrackDetectionsDataset(conf.soccernet_test_path, conf.saved_detections_test_path)
        self.soccernet_val_dataset = TrackDetectionsDataset(conf.soccernet_valid_path, conf.saved_detections_valid_path)
        
        self.registry = {
            # 'soccernet_train': self.soccernet_train_dataset,        
            'soccernet_test': self.soccernet_test_dataset,      
            'soccernet_valid': self.soccernet_val_dataset
        }
        
    def get_train_dataset(self):
        train_datasets = ConcatDataset([self.registry[dataset] for dataset in self.conf.train_datasets])
        return train_datasets
    
    def get_valid_dataset(self):
        valid_datasets = [(self.registry[dataset], dataset) for dataset in self.conf.valid_datasets]
        return valid_datasets        

class TrackSolverTrainer(pl.LightningModule):
    def __init__(self, data_conf: DatasetConfiguration, experiments_folder, batch_size = 32, learning_rate = 1e-4):       
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.data_conf = data_conf

        # Load Model
        self.model = CNNFusion(learning_rate)

        self.model = TransFusion(
            learning_rate,
            num_layers=2,
            d_model=256,
            num_heads=4,
            d_ff=1024,
            input_dim=100,
            output_dim=100,
            max_length=20)


        
        self.data_registry = DatasetRegistry(data_conf) 

        # Then dataloaders, using model's transform
        self.train_dataset = self.data_registry.get_train_dataset() 
        self.val_dataset = self.data_registry.get_valid_dataset() 

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle = True)
        self.val_loaders = [DataLoader(d, batch_size=self.batch_size,  shuffle = True) for d, name in self.val_dataset]
        self.val_loaders_names = [name for d, name in self.val_dataset]
        
        self.experiment_id = None
        self.root_experiments_folder = experiments_folder

    def get_id(self):            
        return self.experiment_id

        
    def setup(self, stage):
        # Define experiment id
        if self.logger is not None and isinstance(self.logger, NeptuneLogger):
            # If using neptune, id is that of neptune experiment
            self.experiment_id = self.logger.version
        else:
            self.experiment_id = str(uuid.uuid4())
        self.experiment_folder = Path(self.root_experiments_folder, self.experiment_id)
        self.experiment_folder.mkdir(parents=True, exist_ok=True)
        
        self.logger.experiment["solver"] = str(self.model.__class__.__name__)
        
        self.logger.experiment["dataset/train_length"] = len(self.train_dataset)
        self.logger.experiment["dataset/val_length"] = len(self.val_dataset)
        self.logger.experiment["dataset/conf"] = self.data_conf.asdict()

    def _common_step(self, batch, batch_id):
        x_batch, gt_batch = batch
        pred_y_batch = self.model(x_batch)
        loss = self.model.loss(pred_y_batch, gt_batch)
        # print("pred", pred_y_batch)
        # print("gt_batch", gt_batch)
        return pred_y_batch, loss

    
    def training_step(self, batch, batch_idx):
        # print("train step batch", batch[0].shape, batch[1].shape)
        _, gt_batch = batch
        pred_y_batch, loss = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        pred_labels = self.model.pred_to_label(pred_y_batch)
        
        accuracy = torch.sum(pred_labels == gt_batch) / float(gt_batch.size(0))
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)        
        
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # print("VALID step batch", batch[0].shape, batch[1].shape)
        # Metrics are logged to current dataloader
        dataset_name = self.val_loaders_names[dataloader_idx]
        _, gt_batch = batch
        pred_y_batch, loss = self._common_step(batch, batch_idx)
        self.log(f'val_loss/{dataset_name}', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx = False)
        
        pred_labels = self.model.pred_to_label(pred_y_batch)
        
        accuracy = torch.sum(pred_labels == gt_batch) / float(gt_batch.size(0))
        self.log(f'val_accuracy/{dataset_name}', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, add_dataloader_idx = False)
        
        return loss
    
    def save_model(self):
        models_folder = Path(self.experiment_folder, "models")
        models_folder.mkdir(exist_ok = True, parents = True)
        model_path = Path(models_folder, f'model_epoch_{self.current_epoch}.pt')
        torch.save(self.model.state_dict(), str(model_path))
        
    def on_validation_epoch_end(self):
        self.save_model()
        
    def configure_optimizers(self):
        return self.model.configure_optimizers()


def train(train_datasetpath, valid_datasetpath, test_datasetpath, saved_detections_valid, saved_detections_test, train_datasets, valid_datasets, epochs, batch_size):   
    data_configuration = DatasetConfiguration(        
        soccernet_train_path = train_datasetpath,
        soccernet_test_path = test_datasetpath,
        soccernet_valid_path = valid_datasetpath,        
        saved_detections_valid_path=saved_detections_valid,
        saved_detections_test_path=saved_detections_test,
        train_datasets = train_datasets,
        valid_datasets = valid_datasets        
    )
    experiments_folder = "../../experiments/track_solver"
    in_trainer = TrackSolverTrainer(data_configuration, experiments_folder=experiments_folder, batch_size=batch_size)
    
    
    neptune_logger = NeptuneLogger(  # replace with your own
        project="eidos/GameStateChallenge",  # format "workspace-name/project-name"
        tags=["TrackSolver"],  # optional
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10,
        logger = neptune_logger,
        enable_checkpointing=False
    )
    
    # Train the Trainer
    trainer.fit(
        in_trainer, in_trainer.train_loader, in_trainer.val_loaders)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--train', default="../../data/SoccerNetGS/train", required=False, help='Path to directory with JerseyDataset train split')
    parser.add_argument('-v', '--val', default="../../data/SoccerNetGS/valid", required=False, help='Path to directory with JerseyDataset val split')
    parser.add_argument('-te', '--test', default="../../data/SoccerNetGS/test", required=False, help='Path to directory with JerseyDataset test split')
    parser.add_argument('-sv', '--saved_valid', required=False, help='Path to directory with saved detections for tracks on valid split')
    parser.add_argument('-ste', '--saved_test', required=False,  help='Path to directory with saved detections for tracks on test split')
    parser.add_argument("--train_datasets", nargs="+", type=str, help="List of training datasets")
    parser.add_argument("--valid_datasets", nargs="+", type=str, help="List of valid datasets")

    parser.add_argument('-ep', '--epochs', type=int, default=200, required=False, help='Number of epochs for training')
    parser.add_argument('-tb', '--batch_size', type=int, default=32, required=False, help='Batch size')

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

    train(args['train'], args['val'], args['test'],
          args['saved_valid'], args['saved_test'],
          args['train_datasets'], args['valid_datasets'],
          epochs = args['epochs'], 
          batch_size = args['batch_size'])