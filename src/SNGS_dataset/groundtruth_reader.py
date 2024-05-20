import os
import json
from typing import Any, Optional
import logging
from dataclasses import dataclass, field, asdict
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    video: str
    image_id: str
    track_id: int
    role: str
    team: str
    role_id: int
    jersey: str
    bbox: dict[str, float] = field(default_factory=dict)
    
class GroundTruthReader:
    def __init__(self, dataset_path: str, videos_list: Optional[list[str]] = None) -> None:
        self.dataset_path = dataset_path
        self.videos_list = videos_list
        if not self.videos_list:
            self.videos_list = [p for p in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, p)) and p[:5] == "SNGS-"]
        self.person_categories = [1, 2, 3, 7]  # 1: player, 2: goalkeeper, 3: referee, 7: other
        self.annotations = self.get_annotations()
        self.annotations_df = pd.DataFrame([asdict(ann) for ann in self.annotations])

    def get_annotations(self) -> list[Annotation]:
        annotations = []
        
        video_paths = [os.path.join(self.dataset_path, video) for video in self.videos_list if os.path.isdir(os.path.join(self.dataset_path, video))]
        # uncomment this (and comment previous line) if want to reduce execution time
        # video_paths = []
        # if "train" in self.dataset_path:
        #     video_paths = [os.path.join(self.dataset_path, "SNGS-060"), os.path.join(self.dataset_path, "SNGS-097"), os.path.join(self.dataset_path, "SNGS-151")]
        # elif "valid" in self.dataset_path:
        #     video_paths = [os.path.join(self.dataset_path, "SNGS-036"), os.path.join(self.dataset_path, "SNGS-039"), os.path.join(self.dataset_path, "SNGS-078")]
        # elif "test" in self.dataset_path:
        #     video_paths = [os.path.join(self.dataset_path, "SNGS-116"), os.path.join(self.dataset_path, "SNGS-133"), os.path.join(self.dataset_path, "SNGS-187")]

        def process_video(video_path):
            video_annotations = []
            try:
                labels_path = os.path.join(video_path, "Labels-GameState.json")
                with open(labels_path) as file:
                    labels_data = json.load(file)
            except FileNotFoundError:
                logger.info(f"Labels file not found for video {os.path.basename(video_path)} at {labels_path}")
                return video_annotations

            for annotation in labels_data["annotations"]:
                if annotation["category_id"] not in self.person_categories:
                    continue
                role_id = map_role_id(annotation)
                video_annotations.append(Annotation(annotation_id=annotation["id"],
                                                   video=os.path.basename(video_path),
                                                   image_id=annotation["image_id"],
                                                   track_id=annotation["track_id"],
                                                   role=annotation["attributes"]["role"],
                                                   team=annotation["attributes"]["team"] if annotation["attributes"]["team"] else "",
                                                   role_id=role_id,
                                                   jersey=annotation["attributes"]["jersey"] if annotation["attributes"]["jersey"] else "",
                                                   bbox={
                                                        "x": annotation["bbox_image"]["x"],
                                                        "y": annotation["bbox_image"]["y"],
                                                        "w": annotation["bbox_image"]["w"],
                                                        "h": annotation["bbox_image"]["h"],
                                                        "x_pitch": annotation["bbox_pitch"]["x_bottom_middle"] if annotation["bbox_pitch"] else None,
                                                        "y_pitch": annotation["bbox_pitch"]["y_bottom_middle"] if annotation["bbox_pitch"] else None
                                                    }))
            return video_annotations

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_video = {executor.submit(process_video, video_path): video_path for video_path in video_paths}
            
            # Process the futures as they complete
            for future in tqdm(as_completed(future_to_video), total=len(future_to_video), desc='Getting annotations', unit='video'):
                video_annotations = future.result()
                annotations.extend(video_annotations)
                
        return annotations

    def get_annotations_from_video(self, video: str) -> list[Annotation]:
        video_annotations = []
        for annotation in self.annotations:
            if annotation.video == video:
                video_annotations.append(annotation)
        return video_annotations
    

if __name__ == "__main__":
    gt_reader = GroundTruthReader("../../data/SoccerNetGS/test", ["SNGS-116", "SNGS-132", "SNGS-187"])
    labels = gt_reader.get_annotations()
    print(labels[0])
