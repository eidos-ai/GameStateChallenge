import os
import json
import argparse
from PIL import Image
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import copy
import psutil
import re

@dataclass
class BBox:
    image_id: str
    image_filename: str
    frame: int
    x: float
    y: float
    w: float
    h: float

@dataclass
class Track:
    track_id: int
    jersey: int
    role_id: int
    video: str
    bboxes: List[BBox]

class TrackJerseyNumberDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.data = []
        self.cache = {"video": None, "images": None}
        
        dirs = [dir_name for dir_name in os.listdir(dataset_folder) if "SNGS" in dir_name]
        
        for dir_name in tqdm(dirs, desc="Processing games", total=len(dirs)):
            json_file_path = os.path.join(dataset_folder, dir_name, "Labels-GameState.json")
            img_folder = os.path.join(dataset_folder, dir_name, "img1")
            if os.path.exists(json_file_path) and os.path.exists(img_folder):
                with open(json_file_path, "r") as f:
                    json_data = json.load(f)
                    images = {image["image_id"]: image["file_name"] for image in json_data["images"]}
                    tracks = {}
                    for annotation in json_data["annotations"]:
                        if "attributes" in annotation and (annotation["attributes"]["role"] == "player" or annotation["attributes"]["role"] == "goalkeeper"):
                            # Get Track or create a new one
                            track_id = annotation["track_id"]
                            track = tracks.get(track_id)
                            if not track:
                                jersey_number = annotation["attributes"]["jersey"]
                                if jersey_number is None:
                                    jersey_number = 0
                                track = Track(
                                    track_id=track_id,
                                    jersey=jersey_number,
                                    role_id=annotation["attributes"]["role"],
                                    video=dir_name,
                                    bboxes=[]
                                )
                                tracks[track_id] = track
                            
                            # Add the Bbox of the current annotation to the Track
                            image_id = annotation["image_id"]
                            image_name = images.get(image_id)
                            frame = int(image_id) % 1000
                            bbox = BBox(
                                image_id=image_id,
                                image_filename=image_name,
                                frame=frame,
                                x=annotation["bbox_image"]["x"],
                                y=annotation["bbox_image"]["y"],
                                w=annotation["bbox_image"]["w"],
                                h=annotation["bbox_image"]["h"]
                            )
                            track.bboxes.append(bbox)
                    self.data.extend(tracks.values())
        
        # Makes iterating over the dataset faster
        self.data = sorted(self.data, key=lambda x: x.video)
  

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        track = self.data[idx]
        
        # Update the cached game images
        if self.cache["video"] != track.video:
            self.cache["images"] = self.load_images_in_video(track.video)
            self.cache["video"] = track.video
            # Print virtual memory usage after loading images
            # process = psutil.Process()
            # virtual_memory_used_bytes = process.memory_info().vms
            # virtual_memory_used_gb = virtual_memory_used_bytes / (1024 ** 3)
            # print(f"Virtual memory used: {virtual_memory_used_gb:.2f} GB")
            
        
        cropped_images = []
        for bbox in track.bboxes:
            image_id = bbox.image_id
            image = self.cache["images"].get(image_id)
            if image is None:
                # This should not happen, all images should be in the cache
                image_path = os.path.join(self.dataset_folder, track.video, "img1", bbox.image_filename)
                image = Image.open(image_path)
            cropped_image = image.crop((bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h))
            if self.transform:
                cropped_image = self.transform(cropped_image)
            cropped_images.append(cropped_image)
        return track, cropped_images
    
    
    # Loads into memory all the images for a game
    def load_images_in_video(self, video_folder):
        images = {}
        img_folder = os.path.join(self.dataset_folder, video_folder, "img1")
        json_file_path = os.path.join(self.dataset_folder, video_folder, "Labels-GameState.json")

        if not os.path.exists(img_folder) or not os.path.exists(json_file_path):
            return images

        with open(json_file_path, "r") as f:
            json_data = json.load(f)
            for image_data in json_data.get("images", []):
                image_id = image_data["image_id"]
                image_file_name = image_data["file_name"]
                image_path = os.path.join(img_folder, image_file_name)
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    images[image_id] = image
        return images
                      

def main():
    parser = argparse.ArgumentParser(description="DataLoader for JerseyNumberDataset.")
    parser.add_argument(
        "--path", type=str, default='../../data/SoccerNetGS/valid', help="Path to the dataset folder"
    )
    args = parser.parse_args()

    # Validate if dataset folder exists
    if not os.path.exists(args.path):
        print("Error: Dataset folder does not exist.")
        return

    dataset = TrackJerseyNumberDataset(
        dataset_folder=args.path
    )

    # Iterate over the dataset using tqdm
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]


if __name__ == "__main__":
    main()
