import os
import json
import logging
from typing import Any
from dataclasses import dataclass, field, asdict
import pandas as pd

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
    def __init__(self, dataset_path: str, videos_list: list[str]) -> None:
        self.dataset_path = dataset_path
        self.videos_list = videos_list
        self.person_categories = [1, 2, 3, 7]  # 1: player, 2: goalkeeper, 3: referee, 7: other
        self.annotations = self.get_annotations()
        self.annotations_df = pd.DataFrame([asdict(ann) for ann in self.annotations])

    def get_annotations(self) -> list[Annotation]:
        annotations = []
        for video in os.listdir(self.dataset_path):
            if self.videos_list and video not in self.videos_list:
                continue
            video_path = os.path.join(self.dataset_path, video)
            if not os.path.isdir(video_path):
                continue
            try:
                labels_path = os.path.join(video_path, "Labels-GameState.json")
                labels_data = json.load(
                    open(labels_path)
                )
            except FileNotFoundError:
                logger.info(f"Labels file not found {labels_path}")
                continue
            for annotation in labels_data["annotations"]:
                if annotation["category_id"] not in self.person_categories:
                    continue
                role_id = map_role_id(annotation)
                annotations.append(Annotation(annotation_id=annotation["id"],
                                               video=video,
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
                                               }))
        return annotations
    
    def get_annotations_from_video(self, video: str) -> list[dict[str, Any]]:
        video_annotations = []
        for annotation in self.annotations:
            if annotation.video == video:
                video_annotations.append(annotation)
        return video_annotations

if __name__ == "__main__":
    gt_reader = GroundTruthReader("../../data/SoccerNetGS/test", ["SNGS-116", "SNGS-132", "SNGS-187"])
    labels = gt_reader.get_annotations()
    print(labels[0])
