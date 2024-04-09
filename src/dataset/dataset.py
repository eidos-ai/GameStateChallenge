import os
import json
import time
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from typing import Optional, Callable
from PIL import Image


class SoccerNetGSDataset(Dataset):

    DATASET_PATH = "data/SoccerNetGS/gamestate-2024/"

    def __init__(
        self,
        videos_directory: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.videos_directory_path = f"{self.DATASET_PATH}{videos_directory}"
        self.transform = transform
        self.target_transform = target_transform

        person_categories = [
            1,
            2,
            3,
            7,
        ]  # 1: player, 2: goalkeeper, 3: referee, 7: other
        self.labels = []
        for video in os.listdir(self.videos_directory_path):
            if os.path.isdir(os.path.join(self.videos_directory_path, video)):
                video_path = f"{self.videos_directory_path}/{video}"
                labels_data = json.load(open(f"{video_path}/Labels-GameState.json"))
                for annotation in labels_data["annotations"]:
                    if annotation["category_id"] in person_categories:
                        self.labels.append(
                            {
                                "annotation_id": annotation["id"],
                                "video": video,
                                "image_id": annotation["image_id"],
                                "track_id": annotation["track_id"],
                                "role": annotation["attributes"]["role"],
                                "bbox": {
                                    "x": annotation["bbox_image"]["x"],
                                    "y": annotation["bbox_image"]["y"],
                                    "w": annotation["bbox_image"]["w"],
                                    "h": annotation["bbox_image"]["h"],
                                },
                            }
                        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        label = self.labels[idx]
        img_path = f"{self.videos_directory_path}/{label['video']}/img1/{label['image_id'][-6:]}.jpg"
        pil_img = Image.open(img_path).convert("RGB")
        left = label["bbox"]["x"]
        top = label["bbox"]["y"]
        right = label["bbox"]["x"] + label["bbox"]["w"]
        bottom = label["bbox"]["y"] + label["bbox"]["h"]
        img_crop = pil_img.crop((left, top, right, bottom))

        if self.transform:
            img_crop = self.transform(img_crop)
        if self.target_transform:
            label = self.target_transform(label)

        return img_crop, label
