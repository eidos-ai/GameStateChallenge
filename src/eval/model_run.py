import torch
import sys
import os
import numpy as np
sys.path.append("..")
sys.path.append("../models")
sys.path.append("../SNGS_dataset")
from tqdm import tqdm
from typing import Any
from models.resnet import ResnetMLP
from models.resnet import TRANSFORM
from collections import defaultdict
from torch.utils.data import DataLoader
from SNGS_dataset.dataset import SoccerNetGSDataset
from SNGS_dataset.sampler import SoccerNetGSBatchSampler


def model_run(loader, model, device, output_directory):
    embeddings_per_video = defaultdict(lambda: defaultdict(list))
    annotations_per_video = defaultdict(lambda: defaultdict(list))
    # No need to calculate gradients as it's an eval
    with torch.no_grad():
        # Iterate over every batch
        for batch_idx, (images, labels) in tqdm(
            enumerate(loader),
            desc="Generating embeddings from every batch",
            total=len(loader),
        ):
            # Get embeddings from batch of images
            images = images.to(device)
            embeddings = model(images)
            embeddings_array = embeddings.cpu().numpy()
            # Iterate within each batch
            batch_size = images.shape[0]
            for batch_element_idx in range(batch_size):
                video = labels["video"][batch_element_idx]
                if embeddings_per_video[video]["annotation_ids"] is None:
                    embeddings_per_video[video]["annotation_ids"] = []
                embeddings_per_video[video]["annotation_ids"].append(labels["annotation_id"][batch_element_idx])
                if embeddings_per_video[video]["embeddings"] is None:
                    embeddings_per_video[video]["embeddings"] = []
                embeddings_per_video[video]["embeddings"].append(embeddings_array[batch_element_idx])

    os.makedirs(output_directory, exist_ok=True)
    for video, data in embeddings_per_video.items():
        annotation_ids = np.array(data["annotation_ids"])
        embeddings = np.array(data["embeddings"])
        output_path = os.path.join(output_directory, f"{video}_embeddings.npz")
        np.savez(output_path, annotation_ids=annotation_ids, embeddings=embeddings)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script for model inference.")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to the model used for inference",
    )
    parser.add_argument(
        "--backbone",
        required=False,
        type=str,
        default="resnet18",
        help="Resnet arch to use as backbone",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        required=False,
        nargs="+",
        default=[],
        help="Hidden layers for the MLP",
    )
    parser.add_argument(
        "--mlp_output_size",
        type=int,
        required=False,
        default=64,
        help="Output features size for the MLP",
    )
    parser.add_argument(
        "--games_list",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="List of games to use for inference",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="../../data/SoccerNetGS/train",
        help="Path to the data",
    )
    parser.add_argument(
        "--n_videos",
        type=int,
        required=False,
        default=1,
        help="Number of different videos per batch",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        required=False,
        default=99,
        help="Number of samples per video per batch",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=4,
        help="Number of workers for the DataLoader",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="./video_files",
        help="Path to save the embeddings",
    )

    args = vars(parser.parse_args())

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")

    model_path = args["model_path"]
    backbone = args["backbone"]
    hidden_layers = args["hidden_layers"]
    mlp_output_size = args["mlp_output_size"]
    games_list = args["games_list"]
    data_path = args["data_path"]
    n_videos_batch = args["n_videos"]
    n_samples_video_batch = args["n_samples"]
    num_workers = args["num_workers"]
    output_path = args["output_path"]
    
    # Load model
    model = ResnetMLP(
        resnet_model=backbone,
        mlp_hidden_layers=hidden_layers,
        mlp_output_size=mlp_output_size,
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Load data
    dataset = SoccerNetGSDataset(dataset_path=data_path, games_list=games_list, transform=TRANSFORM)
    sampler = SoccerNetGSBatchSampler(dataset.labels, n_videos=n_videos_batch, n_samples=n_samples_video_batch)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)

    model_run(loader, model, device, output_path)
