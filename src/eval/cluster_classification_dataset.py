import os
import sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from SNGS_dataset.groundtruth_reader import GroundTruthReader
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict
from torch.utils.data import Dataset


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


@dataclass
class ClusterScenario:
    this_tracks_role_id: int
    this_tracks: List[Track]
    rest_tracks: List[Track]
    def __str__(self):
        return f'ClusterScenario(this_tracks_role_id={self.this_tracks_role_id}, this_tracks=(lenght: {len(self.this_tracks)}), rest_tracks=(lenght: {len(self.rest_tracks)}))'
    def __repr__(self):
        return self.__str__()


def get_tracks(videos_list: List[str], gt_reader: GroundTruthReader) -> Dict[str, List[Track]]:
    total_tracks = defaultdict(list)
    for video in videos_list:
        video_tracks = []
        video_annotations = gt_reader.get_annotations_from_video(video)
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
            # if bbox.x_pitch is None or bbox.y_pitch is None:
            #     print(annotation)

            for track in video_tracks:
                if track.track_id == annotation.track_id:
                    track.bboxes.append(bbox)
        total_tracks[video] = video_tracks
    return total_tracks


class ClusterClassificationDataset(Dataset):
    def __init__(self, data_path: str, videos_list: List[str]) -> None:
        self.data_path = data_path
        self.videos_list = videos_list if videos_list else os.listdir(data_path)
        self.gt_reader = GroundTruthReader(self.data_path, self.videos_list)
        self.tracks = get_tracks(self.videos_list, self.gt_reader)
        self.cluster_scenarios = []
        for video in self.videos_list:
            role_id_in_video = set()
            for track in self.tracks[video]:
                role_id_in_video.add(track.role_id)
            for role_id in role_id_in_video:
                this_tracks = []
                rest_tracks = []
                for track in self.tracks[video]:
                    if track.role_id == role_id:
                        this_tracks.append(track)
                    else:
                        rest_tracks.append(track)
                cluster_scenario = ClusterScenario(this_tracks_role_id=role_id, this_tracks=this_tracks, rest_tracks=rest_tracks)
                self.cluster_scenarios.append(cluster_scenario)

    def __len__(self) -> int:
        return len(self.cluster_scenarios)

    def __getitem__(self, idx: int) -> ClusterScenario:
        return self.cluster_scenarios[idx], self.cluster_scenarios[idx].this_tracks_role_id


if __name__ == "__main__":
    data_path = "../../data/SoccerNetGS/valid"
    videos_list = ["SNGS-021", "SNGS-039", "SNGS-078"]
    dataset = ClusterClassificationDataset(data_path, videos_list)

    for i in range(len(dataset)):
        print(dataset[i])
