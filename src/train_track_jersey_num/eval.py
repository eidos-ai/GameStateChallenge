import os
import sys
import logging
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score
import logging


parent_folder = '/'.join(__file__.split('/')[:-2])
sys.path.append(parent_folder)

from gsr.utils.neptune_logger import NeptuneLogger
from train_jersey_num.models.convnext import ConvNext
# from bbox_dataset import BBoxDataset, bbox_collate_fn
from dataloader import TrackJerseyNumberDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MaxMeanTrackSolver(nn.Module):
    def __init__(self):
        super(MaxMeanTrackSolver, self).__init__()

    def forward(self, track_predictions: np.array):
        mean = np.mean(track_predictions, axis= 0)
        max_label = np.argmax(mean)
        return max_label

def eval(data_path, saved_path, enable_neptune, exp_name):
    # Dataset
    split = data_path.split("/")[-1]
    dataset = TrackJerseyNumberDataset(data_path, return_img= False)
    track_solver = MaxMeanTrackSolver()

    neptune_logger = NeptuneLogger(enable_run=enable_neptune, project="eidos/GameStateChallenge", run_name_or_id=exp_name)
    neptune_logger.tags(["TrackSolver"])
    neptune_logger.param("solver", track_solver.__class__.__name__)
    neptune_logger.param("detections_path", saved_path)
    
    # Iterate over each saved track
    labels, preds = [], []
    for track, _ in tqdm(dataset, desc="Solving tracks", total=len(dataset)):
        file_path = os.path.join(saved_path, f"{track.video}_{track.track_id}.npz")
        saved_detections = np.load(file_path)
        
        # Access data
        track_id = saved_detections['track_id']
        video = saved_detections['video']
        track_predictions = saved_detections['track_predictions']
        # Reshape becasue of bug on save
        track_predictions = track_predictions.reshape(-1,100)
        
        predicted_track = track_solver(track_predictions)
        labels.append(track.jersey)
        preds.append(predicted_track)
    labels = np.array(labels)
    preds = np.array(preds)
    
    # Accuracy
    acc = accuracy_score(labels,preds)
    neptune_logger.log(f"training/val_accuracy/soccernet_{split}", acc)
    logging.info(f"Accuracy on {split}: {acc}")

    # Acc for non-zero tracks
    non_zero_indices = labels != 0
    accuracy_non_zero = accuracy_score(labels[non_zero_indices], preds[non_zero_indices])
    neptune_logger.log(f"evaluation/{split}/nonzero-acc", accuracy_non_zero)
    logging.info(f"Nonzero accuracy on {split}: {accuracy_non_zero}")
            
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classification model.")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="../../data/SoccerNetGS/valid",
        help="Path to the dataset.",
    )   
    parser.add_argument(
        "--saved_detections",
        type=str,
        required=True,
        help="Path to stored model outs.",
    )
    parser.add_argument("--neptune", required=False, action="store_true", default=False, help="Log experiment results to neptune")
    parser.add_argument("--name", required=False, type=str, help="The name of the experiment")

    args = vars(parser.parse_args())
    data_path = args["data_path"]    
    saved_path = args["saved_detections"]
    enable_neptune = args["neptune"]
    exp_name = args["name"]
    eval(data_path, saved_path, enable_neptune, exp_name)
    