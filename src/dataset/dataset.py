import os
import json
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset


class SoccerNetGSDataset(Dataset):

    DATASET_PATH = "data/SoccerNetGS/gamestate-2024/"

    def __init__(
        self, videos_directory: str, transform=None, target_transform=None
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
        labels_list = []
        for video in os.listdir(self.videos_directory_path):
            if os.path.isdir(os.path.join(self.videos_directory_path, video)):
                video_path = f"{self.videos_directory_path}/{video}"
                labels_data = json.load(open(f"{video_path}/Labels-GameState.json"))
                for annotation in labels_data["annotations"]:
                    if annotation["category_id"] in person_categories:
                        pass
                        labels_list.append(
                            {
                                "video": video,
                                "image": annotation["image_id"][-6:],
                                "role": annotation["attributes"]["role"],
                                "label_id": annotation["id"],
                                "track_id": annotation["track_id"],
                                "bbox": {
                                    "x": annotation["bbox_image"]["x"],
                                    "y": annotation["bbox_image"]["y"],
                                    "w": annotation["bbox_image"]["w"],
                                    "h": annotation["bbox_image"]["h"],
                                },
                                "crop_img": None,
                            }
                        )
        self.labels = pd.DataFrame(labels_list)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple:
        label = self.labels.iloc[idx]
        img_path = f"{self.videos_directory_path}/{label.video}/img1/{label.image}.jpg"
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    dataset = SoccerNetGSDataset("train")
    print(dataset.__len__())
    print(dataset.__getitem__(0))
