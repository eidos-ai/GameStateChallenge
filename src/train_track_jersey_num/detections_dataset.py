from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np

from dataloader import TrackJerseyNumberDataset

class TrackDetectionsDataset(Dataset):
    def __init__(self, dataset_folder, saved_detections_path):
        tracks_dataset = TrackJerseyNumberDataset(dataset_folder, return_img= False)

        self.track_files = []
        for track, _ in tqdm(tracks_dataset, desc="Loading tracks detections", total=len(tracks_dataset)):
            file_path = os.path.join(saved_detections_path, f"{track.video}_{track.track_id}.npz")
            if Path(file_path).exists():
                self.track_files.append([file_path, track])

    def __len__(self):
        return len(self.track_files)
    

    def __getitem__(self, idx):
        file_path, track = self.track_files[idx]
        saved_detections = np.load(file_path)
        
        track_id = saved_detections['track_id']
        video = saved_detections['video']
        track_predictions = saved_detections['track_predictions']
        
        # Reshape becasue of bug on save
        track_predictions = track_predictions.reshape(-1,100)

        # Returning mean
        track_average_scores = np.mean(track_predictions, axis= 0)
        track_average_scores = np.expand_dims(track_average_scores, axis=0)
        return track_average_scores, track.jersey 
