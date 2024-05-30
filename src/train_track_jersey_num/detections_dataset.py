from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import os
import torch
import numpy as np
import scipy
from torchvision import transforms

from dataloader import TrackJerseyNumberDataset

def average_split(arr, num_splits=15):
    arr = torch.from_numpy(arr)
    # Calculate the size of each full split and the number of elements in the last, possibly smaller, split
    split_size = arr.shape[0] // num_splits
    remainder = arr.shape[0] % num_splits

    # Initialize an array to hold each averaged and max value for each segment
    results = torch.zeros((num_splits, 200), dtype=torch.float32)  # Explicitly setting dtype to torch.float32

    # Process each full-sized segment
    for i in range(num_splits):
        start_idx = i * split_size
        if i == num_splits - 1:
            # Handle the last segment which might be larger due to the remainder
            end_idx = start_idx + split_size + remainder
        else:
            end_idx = start_idx + split_size

        # Check if the segment has elements
        if start_idx < end_idx:  # This ensures the segment is not empty
            # Calculate the average and max of the segment
            segment_avg = arr[start_idx:end_idx].mean(dim=0)
            segment_max = arr[start_idx:end_idx].max(dim=0).values
            results[i, :100] = segment_avg  # Store averages in the first 100 positions
            results[i, 100:] = segment_max  # Store maxima in the next 100 positions
        else:
            # If no elements in the segment, fill with zeros or another placeholder value
            results[i, :] = torch.zeros(200, dtype=torch.float32)

    return results


class FlipSignal:
    def __init__(self, probability=0.5):
        self.probability = probability
    
    def __call__(self, sample):
        if np.random.rand() >= self.probability:
            return sample  # Return the original sample without augmentation

        X, original_label = sample
        np.random.seed()  # Reset seed for random label generation
        d, num_signals = X.shape

        # Generate a new random label different from the original
        copy_label = np.random.randint(0, 100)
        while copy_label == original_label:
            copy_label = np.random.randint(0, 100)

        original_label_digits = set(str(original_label))
        original_neighbors = [i for i in range(num_signals) if set(str(i)) & original_label_digits]

        copy_label_digits = set(str(copy_label))
        copy_neighbors = [i for i in range(num_signals) if set(str(i)) & copy_label_digits]

        np.random.shuffle(original_neighbors)
        np.random.shuffle(copy_neighbors)

        mapping = {}
        used_indices = set()
        min_length = min(len(original_neighbors), len(copy_neighbors))
        for i in range(min_length):
            orig_idx = original_neighbors[i]
            copy_idx = copy_neighbors[i]
            if copy_idx not in used_indices:
                mapping[orig_idx] = copy_idx
                used_indices.add(copy_idx)

        new_X = np.zeros_like(X)
        new_X[:, copy_label] = X[:, original_label]

        for orig_idx, mapped_idx in mapping.items():
            new_X[:, mapped_idx] = X[:, orig_idx]

        remaining_indices = list(set(range(num_signals)) - set(new_X.any(axis=0).nonzero()[0]))
        shuffle_indices = np.random.choice(remaining_indices, size=len(remaining_indices), replace=False)
        
        for idx, new_idx in zip(remaining_indices, shuffle_indices):
            new_X[:, idx] = X[:, new_idx]

        return new_X, copy_label

class TrackDetectionsDataset(Dataset):
    def __init__(self, dataset_folder, saved_detections_path, n_samples, augment = False):
        tracks_dataset = TrackJerseyNumberDataset(dataset_folder, return_img= False)
        self.n_samples = n_samples
        self.track_files = []
        for track, _ in tqdm(tracks_dataset, desc="Loading tracks detections", total=len(tracks_dataset)):
            file_path = os.path.join(saved_detections_path, f"{track.video}_{track.track_id}.npz")
            if Path(file_path).exists():
                self.track_files.append([file_path, track])

        if augment:
            self.transform = transforms.Compose([
                FlipSignal(probability=0.8)
            ])
        else:
            self.transform = None
            
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
        jersey_number = track.jersey

        # Apply the transform if available
        if self.transform:
            track_predictions, jersey_number = self.transform((track_predictions, jersey_number))

        track_predictions = scipy.special.softmax(track_predictions,axis=0)
        
        # Returning mean
        # track_average_scores = np.mean(track_predictions, axis= 0)
        # track_average_scores = np.expand_dims(track_average_scores, axis=0)

        # Returning sample mean
        track_sample_scores = average_split(track_predictions, num_splits = self.n_samples)
        if np.isnan(track_sample_scores).any():
            return self.__getitem__(idx-1)
        return track_sample_scores, jersey_number
