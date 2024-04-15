import os
import json
from torch.utils.data import Dataset
from typing import Optional, Callable
from PIL import Image
import random

LABELS_MAP = {
    "player_left": 0,
    "player_right": 1,
    "goalkeeper_left": 2,
    "goalkeeper_right": 3,
    "referee": 4,
    "other": 5,
}


def map_role_id(annotation: dict) -> int:
    role_id = None
    role = annotation["attributes"]["role"]
    if role == "player":
        role_id = (
            LABELS_MAP["player_left"]
            if annotation["attributes"]["team"] == "left"
            else LABELS_MAP["player_right"]
        )
    elif role == "goalkeeper":
        role_id = (
            LABELS_MAP["goalkeeper_left"]
            if annotation["attributes"]["team"] == "left"
            else LABELS_MAP["goalkeeper_right"]
        )
    elif role == "referee":
        role_id = LABELS_MAP["referee"]
    elif role == "other":
        role_id = LABELS_MAP["other"]
    else:
        role_id = -1
    return role_id


class SoccerNetGSDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
        games_list: Optional[list] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform

        person_categories = [
            1,
            2,
            3,
            7,
        ]  # 1: player, 2: goalkeeper, 3: referee, 7: other
        self.labels = []
        for video in os.listdir(self.dataset_path):
            if games_list and video not in games_list:
                continue
            video_path = os.path.join(self.dataset_path, video)
            if not os.path.isdir(video_path):
                continue
            try:
                labels_data = json.load(
                    open(os.path.join(video_path, "Labels-GameState.json"))
                )
            except FileNotFoundError:
                continue
            for annotation in labels_data["annotations"]:
                if annotation["category_id"] not in person_categories:
                    continue
                role_id = map_role_id(annotation)
                self.labels.append(
                    {
                        "annotation_id": annotation["id"],
                        "video": video,
                        "image_id": annotation["image_id"],
                        "track_id": annotation["track_id"],
                        "role": annotation["attributes"]["role"],
                        "team": (
                            annotation["attributes"]["team"]
                            if annotation["attributes"]["team"]
                            else ""
                        ),
                        "role_id": role_id,
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
        img_path = os.path.join(
            self.dataset_path, label["video"], "img1", f"{label['image_id'][-6:]}.jpg"
        )
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


if __name__ == "__main__":
    import sys
    from tqdm import tqdm

    sys.path.append("..")
    sys.path.append("../models")
    # sys.path.append("/Users/francisco/Documents/GameStateChallenge")
    # sys.path.append("/Users/francisco/Documents/GameStateChallenge/src")
    # sys.path.append("/Users/francisco/Documents/GameStateChallenge/src/models")
    from models.resnet import TRANSFORM

    dataset = SoccerNetGSDataset(
        "../../data/SoccerNetGS/train",
        games_list=["SNGS-175", "SNGS-111", "SNGS-050", "SNGS-077"],
        transform=TRANSFORM,
    )

    # dataset = SoccerNetGSDataset(
    #     "data/SoccerNetGS/train",
    #     games_list=["SNGS-175", "SNGS-111", "SNGS-050", "SNGS-077"],
    #     transform=TRANSFORM,
    # )

    print(f"Dataset lenght: {len(dataset)}")
    idx_th = 100000
    for idx, dataset_item in tqdm(enumerate(dataset), total=len(dataset)):
        crop, label = dataset_item
        if label["video"] != "SNGS-111" and label["video"] != "SNGS-077":
            print("error in video name", label["video"])

        if idx == idx_th:
            break
