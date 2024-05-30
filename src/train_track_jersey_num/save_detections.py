import os
import sys
import logging
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm

parent_folder = '/'.join(__file__.split('/')[:-2])
sys.path.append(parent_folder)

from train_jersey_num.models.convnext import ConvNext
# from bbox_dataset import BBoxDataset, bbox_collate_fn
from dataloader import TrackJerseyNumberDataset

def images_to_batches(image_list, batch_size, inference_transform):
    transform = inference_transform
    batches = []

    for i in range(0, len(image_list), batch_size):
        batch = torch.stack([transform(image) for image in image_list[i:i+batch_size]])
        batches.append(batch)
    
    return batches

def evaluate_track_number_recognition(data_path, model_path, out_path):
    device = "cuda"
    Path(out_path).mkdir(exist_ok=True, parents=True)
    
    # Load model
    model_name = "convnextv2_large.fcmae_ft_in22k_in1k"
    model = ConvNext(model_name, 100, None)
    model.load_model(model_path)
    model = model.to(device)
    inference_transform = model.get_transform(is_training= False)
    
     # Load tracks dataset
    dataset = TrackJerseyNumberDataset(data_path)
    # Iter over dataset and save detections for each track on video
    for track, images in tqdm(dataset, desc="Processing videos", total=len(dataset)):
        batches = images_to_batches(images, batch_size=8, inference_transform=inference_transform)
        batches_predictions = []
        for batch in batches:
            batch = batch.to(device)
            batch_prediction = model(batch)
            batches_predictions.extend(batch_prediction.detach().cpu().numpy())

        track_predictions = np.vstack(batches_predictions)
        print(track_predictions.shape)
        np.savez(f'{out_path}/{track.video}_{track.track_id}', 
                 track_id=track.track_id,
                 video=track.video,
                 track_predictions=track_predictions)
            
    
    
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
        "--model_path",
        type=str,
        required=True,
        help="Path to the model.",
    )   
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to the model.",
    )   
    args = vars(parser.parse_args())
    data_path = args["data_path"]    
    model_path = args["model_path"]
    out_path = args["out_path"]
    evaluate_track_number_recognition(data_path, model_path, out_path)