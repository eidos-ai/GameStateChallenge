import os
from torch.utils.data import Dataset
from typing import Optional, Callable
from SNGS_dataset.groundtruth_reader import GroundTruthReader
from PIL import Image
from dataclasses import asdict
from typing import List, Tuple, Dict, Any
from torchvision.transforms import ToTensor
import torch

# This collate_fn is necesary to avoid problems due to None values on some labels
def soccernet_dataset_collate_fn(batch: List[Tuple[Any, Dict]]) -> Tuple[List[Any], List[Dict]]:
     # Transform to convert PIL images to tensors
    to_tensor = ToTensor()
    
    # Initialize the images list and a dictionary to hold processed labels
    images = []
    batch_labels = {}

    for img, label in batch:
        # Convert image to tensor if it is a PIL Image, otherwise check if it's already a tensor
        images.append(img)

        # Initialize the dictionary structure for batch_labels based on the first label
        if not batch_labels:
            for key in label.keys():
                batch_labels[key] = []

        # Append each attribute of the label to the corresponding list in batch_labels
        for key in label.keys():
            if key == 'bbox':
                # Special handling for 'bbox' to ensure no None values and correct structure
                bbox_dict = {sub_key: label['bbox'].get(sub_key, 0) if label['bbox'] else 0 for sub_key in ['x', 'y', 'w', 'h', 'x_pitch', 'y_pitch']}
                batch_labels['bbox'].append(bbox_dict)
            else:
                batch_labels[key].append(label.get(key, ''))

    # Convert images list to a tensor
    images = torch.stack(images, dim=0)
    
    return images, batch_labels
    
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
