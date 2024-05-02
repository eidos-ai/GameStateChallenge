import os
from torch.utils.data import Dataset
from typing import Optional, Callable
from SNGS_dataset.groundtruth_reader import GroundTruthReader
from PIL import Image
from dataclasses import asdict


class SoccerNetGSDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        videos_list: Optional[list] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.transform = transform
        self.target_transform = target_transform
        self.labels = []
        self.videos_list = videos_list

        gt_reader = GroundTruthReader(dataset_path, videos_list)
        for annotation in gt_reader.annotations: # returns list of Annotation objects
            annotation_dict = asdict(annotation)
            self.labels.append(annotation_dict)
            
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
    from models.resnet import TRANSFORM

    videos = ["SNGS-116", "SNGS-132", "SNGS-187"]

    dataset = SoccerNetGSDataset(
        "../../data/SoccerNetGS/test",
        videos_list=videos,
        transform=TRANSFORM,
    )

    print(f"Dataset lenght: {len(dataset)}")
    idx_th = 10000
    for idx, dataset_item in tqdm(enumerate(dataset), total=len(dataset)):
        crop, label = dataset_item
        if label["video"] not in videos:
            print("error in video name", label["video"])

        if idx == idx_th:
            break
