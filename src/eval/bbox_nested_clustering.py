import os
import torch
from tqdm import tqdm
from PIL import Image
from typing import List
from shutil import copy2
from torchvision import transforms
from sklearn.cluster import DBSCAN
from models.resnet import ResnetMLP


def nested_clustering(model: ResnetMLP, transform: transforms, dbscan: DBSCAN, crops_dir: str) -> None:
    crops = os.listdir(crops_dir)

    with torch.no_grad():
        total_embeddings = []
        for crop in tqdm(crops, total=len(crops), desc="Crops iteration"):
            if crop[0] == ".":
                continue
            crop_path = os.path.join(crops_dir, crop)
            pil_img = Image.open(crop_path).convert("RGB")
            pil_img = transform(pil_img).unsqueeze(0)
            img = pil_img.to(device)
            embeddings = model(img)
            embeddings_array = embeddings.cpu().numpy().flatten()
            total_embeddings.append(embeddings_array)
   
        model_output = dbscan.fit(total_embeddings)
        predictions = model_output.labels_
        for i in range(len(predictions)):
            temp_prediction = str(predictions[i]).zfill(3) if predictions[i] != -1 else str(predictions[i])
            prediction_dir = os.path.join(crops_dir, "nested_clustering", temp_prediction)
            os.makedirs(prediction_dir, exist_ok=True)
            crop_path = os.path.join(crops_dir, crops[i])
            try:
                copy2(crop_path, prediction_dir)
            except:
                continue


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
        "--bbox_path",
        type=str,
        required=True, # example: ./bbox_clusters/SNGS-006/001
        help="Path to the bbox crops.",
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
    bbox_path = args["bbox_path"]
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
    nested_clustering(model, TRANSFORM, dbscan, bbox_path)
