import os
import torch
from tqdm import tqdm
from PIL import Image
from typing import List
from shutil import copy2
from torchvision import transforms
from sklearn.cluster import DBSCAN
from models.resnet import ResnetMLP
from collections import defaultdict


def cluster_bboxes_on_video(model: ResnetMLP, data_path: str, bbox_path: str, output_path: str, videos_list: List[str], transform: transforms, dbscan: DBSCAN) -> None:
    if videos_list is None:
        videos_list = os.listdir(data_path)
        videos_list = [item for item in videos_list if item[:5] == "SNGS-"]

    with torch.no_grad():
        model.eval()
        for video in tqdm(videos_list, total=len(videos_list), desc="Videos iteration"):
            frames_directory = os.path.join(bbox_path, video)
            crops = os.listdir(frames_directory)
            for crop in tqdm(crops, total=len(crops), desc=f"Crops of video {video}"):
                if crop[0] == ".":
                    continue
                crop_path = os.path.join(frames_directory, crop)
                pil_img = Image.open(crop_path).convert("RGB")
                pil_img = transform(pil_img).unsqueeze(0)
                img = pil_img.to(device)
                embeddings = model(img)
                embeddings_array = embeddings.cpu().numpy()
                model_output = dbscan.fit(embeddings_array)
                predictions = model_output.labels_
                prediction_dir = os.path.join(output_path, video, str(predictions[0]))
                if predictions[0] != -1:
                    print(predictions[0])
                os.makedirs(prediction_dir, exist_ok=True)
                copy2(crop_path, prediction_dir)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.append("..")
    sys.path.append("../models")
    from models.resnet import TRANSFORM

    parser = argparse.ArgumentParser(description="The script's input should be the output of the bbox_detection.py script.")
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
        default=128,
        help="Output features size for the MLP",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="../../data/SoccerNetGS/challenge",
        help="Path to the data (train, valid, test or challenge).",
    )
    parser.add_argument(
        "--bbox_path",
        type=str,
        required=False,
        default="./bbox_detections",
        help="Path to the bbox crops.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="./bbox_clusters",
        help="Path to export the bbox crops after clustering.",
    )
    parser.add_argument(
        "--videos_list",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="List of videos to use.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        required=False,
        default=0.5,
        help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        required=False,
        default=5,
        help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.",
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
    data_path = args["data_path"]
    bbox_path = args["bbox_path"]
    output_path = args["output_path"]
    videos_list = args["videos_list"]
    eps = args["eps"]
    min_samples = args["min_samples"]

    # Load model
    model = ResnetMLP(
        resnet_model=backbone,
        mlp_hidden_layers=hidden_layers,
        mlp_output_size=mlp_output_size,
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_bboxes_on_video(model, data_path, bbox_path, output_path, videos_list, TRANSFORM, dbscan)
