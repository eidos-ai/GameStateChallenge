import os
import json
from typing import Any
from collections import defaultdict


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

class GroundTruthReader:
    def __init__(self, dataset_path: str, games_list: list[str]) -> None:
        self.dataset_path = dataset_path
        self.games_list = games_list
        self.person_categories = [1, 2, 3, 7]  # 1: player, 2: goalkeeper, 3: referee, 7: other

    def get_annotations(self) -> list[dict]:
        annotations = []
        for video in os.listdir(self.dataset_path):
            if self.games_list and video not in self.games_list:
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
                if annotation["category_id"] not in self.person_categories:
                    continue
                role_id = map_role_id(annotation)
                annotations.append(
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
        return annotations
    
    def get_annotations_from_video(self, video: str) -> list[dict[str, Any]]:
        video_annotations = []
        for annotation in self.get_annotations():
            if annotation["video"] == video:
                video_annotations.append(annotation)
        return video_annotations

if __name__ == "__main__":
    gt_reader = GroundTruthReader("../../data/SoccerNetGS/test", ["SNGS-116", "SNGS-132", "SNGS-187"])
    labels = gt_reader.get_annotations()
    print(labels)
