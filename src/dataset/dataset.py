import os
import json
from torch.utils.data import Dataset
from typing import Optional, Callable
from PIL import Image


class SoccerNetGSDataset(Dataset):

    def __init__(
        self,
        dataset_path: str,
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
            video_path = os.path.join(self.dataset_path, video)
            if not os.path.isdir(video_path):
                continue
            labels_data = json.load(
                open(os.path.join(video_path, "Labels-GameState.json"))
            )
            for annotation in labels_data["annotations"]:
                if annotation["category_id"] not in person_categories:
                    continue
                self.labels.append(
                    {
                        "annotation_id": annotation["id"],
                        "video": video,
                        "image_id": annotation["image_id"],
                        "track_id": annotation["track_id"],
                        "role": annotation["attributes"]["role"],
                        "team": annotation["attributes"]["team"],
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
