import pandas as pd
import numpy as np
import torch
import logging
import sys
import os
import torch

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.videolevel_module import VideoLevelModule
from torchvision import transforms

from gsr.modules.jersey_number.models.samplelstm import SampleLSTM

log = logging.getLogger(__name__)


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
    
class TrackJerseyNumberDetection(VideoLevelModule):
    input_columns = ["bbox_jersey_number"]
    output_columns = ["jersey", "jn_tracklet_confidence"]
    collate_fn = default_collate
    
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.num_splits = cfg.num_splits
        self.model = SampleLSTM(1e-4, self.cfg.hidden_size, self.cfg.num_layers)
        self.model.load_model(cfg.model_path)
        self.model.eval()
        
#         learning_rate = 1e-4 # Not used really
#         self.model = ConvNext(self.cfg.model_name, 100, learning_rate)
#         self.model.load_model(self.cfg.model_path)
#         self.model.to(self.cfg.device)
#         self.inference_transform = self.model.get_transform(is_training= False)
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        df_sorted = detections.sort_values(by=['track_id', 'image_id'])
        # Group by track_id and aggregate jersey_number_detection into a single array
        result_df = df_sorted.groupby('track_id')['bbox_jersey_number'].agg(lambda x: np.vstack(x))
    
        # Process the concatenated array in batches of 8 track_ids
        for i in range(0, len(result_df), self.cfg.batch_size):
            batch_track_ids = list(result_df.keys())[i:i + self.cfg.batch_size]
            batch_track_detections = []

            # Prepare the batch
            for track_id in batch_track_ids:
                combined_array = result_df[track_id]
                track_detections = average_split(combined_array, 10)
                track_detections = track_detections.unsqueeze(dim=0)
                batch_track_detections.append(track_detections)

            # Concatenate all detections in the batch for model processing
            batch_track_detections = torch.cat(batch_track_detections, dim=0)

            # Process the batch through the model
            class_predictions = self.model(batch_track_detections)
            pred_jerseys, pred_confs = self.model.pred_to_label(class_predictions)

            # Print the results
            for idx, track_id in enumerate(batch_track_ids):
                # Update the DataFrame with the predicted jersey numbers and confidences
                track_indices = detections[detections['track_id'] == track_id].index
                detections.loc[track_indices, 'jersey'] = int(pred_jerseys[idx].numpy())
                detections.loc[track_indices, 'jersey_number'] = int(pred_jerseys[idx].numpy())
                detections.loc[track_indices, 'jn_tracklet_confidence'] = pred_confs[idx].numpy()

        return detections