import os
import sys

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from SNGS_dataset.groundtruth_reader import GroundTruthReader
from dataclasses import dataclass
from typing import List
from collections import defaultdict

@dataclass
class BBox:
    frame: int
    x_image: float
    y_image: float
    w_image: float
    h_image: float
    x_pitch: float
    y_pitch: float

@dataclass
class Track:
    track_id: int
    role_id: int
    video: str
    bboxes: List[BBox]

    def __str__(self):
        return f'Track(track_id={self.track_id}, role_id={self.role_id}, video={self.video}, bboxes=(lenght: {len(self.bboxes)}))'

    def __repr__(self):
        return self.__str__()

class ClusterClassificationDataset:
    def __init__(self, data_path: str, videos_list: List[str]):
        self.data_path = data_path
        self.videos_list = videos_list if videos_list else os.listdir(data_path)
        self.gt_reader = GroundTruthReader(self.data_path, self.videos_list)

    def get_tracks(self):
        total_tracks = defaultdict(list)
        for video in self.videos_list:
            video_tracks = []
            video_annotations = self.gt_reader.get_annotations_from_video(video)
            for annotation in video_annotations:
                if annotation.track_id not in [track.track_id for track in video_tracks]:
                    track = Track(track_id=annotation.track_id, role_id=annotation.role_id, video=annotation.video, bboxes=[])
                    video_tracks.append(track)
                bbox = BBox(frame=annotation.image_id,
                            x_image=annotation.bbox["x"],
                            y_image=annotation.bbox["y"],
                            w_image=annotation.bbox["w"],
                            h_image=annotation.bbox["h"],
                            x_pitch=float(round(annotation.bbox["x_pitch"], 4)) if annotation.bbox["x_pitch"] else None, # it is possible for this value to be None
                            y_pitch=float(round(annotation.bbox["y_pitch"], 4)) if annotation.bbox["y_pitch"] else None  # it is possible for this value to be None
                           )
                for track in video_tracks:
                    if track.track_id == annotation.track_id:
                        track.bboxes.append(bbox)
            total_tracks[video] = video_tracks
        return total_tracks

if __name__ == "__main__":
    data_path = "../../data/SoccerNetGS/valid"
    videos_list = ["SNGS-021", "SNGS-039", "SNGS-078"]

    dataset = ClusterClassificationDataset(data_path, videos_list)
    tracks = dataset.get_tracks()

    for video in videos_list:
        for track in tracks[video]:
            print(track)
            print("/////////////////")
            for bbox in track.bboxes:
                print(bbox)
            break
        break
