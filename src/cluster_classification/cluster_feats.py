import numpy as np
from dataclasses import dataclass, astuple, fields
from collections import defaultdict
from typing import List
from cluster_classification_dataset import Track
from gsr.utils.neptune_logger import NeptuneLogger

@dataclass
class Feats:
    x_avg_pos_this_tracks: float | None
    y_avg_pos_this_tracks: float | None
    x_avg_pos_rest_tracks: float | None
    y_avg_pos_rest_tracks: float | None
    len_this_tracks: int
    len_rest_tracks: int
    avg_this_tracks_per_frame: float
    avg_rest_tracks_per_frame: float
    apps_this_tracks_left_half: int
    apps_this_tracks_right_half: int
    apps_rest_tracks_left_half: int
    apps_rest_tracks_right_half: int

    def __str__(self):
        return f'Feats(\n    x_avg_pos_this_tracks={self.x_avg_pos_this_tracks}, \n    y_avg_pos_this_tracks={self.y_avg_pos_this_tracks}, \n    x_avg_pos_rest_tracks={self.x_avg_pos_rest_tracks}, \n    y_avg_pos_rest_tracks={self.y_avg_pos_rest_tracks}, \n    len_this_tracks={self.len_this_tracks}, \n    len_rest_tracks={self.len_rest_tracks}, \n    avg_this_tracks_per_frame={self.avg_this_tracks_per_frame}, \n    avg_rest_tracks_per_frame={self.avg_rest_tracks_per_frame}, \n    apps_this_tracks_left_half={self.apps_this_tracks_left_half}, \n    apps_this_tracks_right_half={self.apps_this_tracks_right_half}, \n    apps_rest_tracks_left_half={self.apps_rest_tracks_left_half}, \n    apps_rest_tracks_right_half={self.apps_rest_tracks_right_half} \n)'
    def __repr__(self):
        return self.__str__()

    def to_vector(self):
        return np.array(astuple(self))


class ClusterFeats():
    def __init__(self, neptune_logger: NeptuneLogger=None):
        self.neptune_logger = neptune_logger
        
    def forward(self, this_tracks: List[Track], rest_tracks: List[Track]) -> Feats:
        x_avg_pos_this_tracks = 0
        y_avg_pos_this_tracks = 0
        x_avg_pos_rest_tracks = 0
        y_avg_pos_rest_tracks = 0

        apps_this_tracks_left_half = 0
        apps_this_tracks_right_half = 0
        apps_rest_tracks_left_half = 0
        apps_rest_tracks_right_half = 0

        len_this_tracks = len(this_tracks)
        len_rest_tracks = len(rest_tracks)

        not_none_bboxes_this_tracks = 0
        not_none_bboxes_rest_tracks = 0

        for track in this_tracks:
            not_none_bboxes_this_tracks += len(track.bboxes)

            for bbox in track.bboxes:
                if bbox.x_pitch is not None or bbox.y_pitch is not None:
                    x_avg_pos_this_tracks += bbox.x_pitch
                    y_avg_pos_this_tracks += bbox.y_pitch
                    if bbox.x_pitch < 0:
                        apps_this_tracks_left_half += 1
                    else:
                        apps_this_tracks_right_half += 1
                else:
                    not_none_bboxes_this_tracks -= 1

        for track in rest_tracks:
            not_none_bboxes_rest_tracks += len(track.bboxes)

            for bbox in track.bboxes:
                if bbox.x_pitch is not None or bbox.y_pitch is not None:
                    x_avg_pos_rest_tracks += bbox.x_pitch
                    y_avg_pos_rest_tracks += bbox.y_pitch
                    if bbox.x_pitch < 0:
                        apps_rest_tracks_left_half += 1
                    else:
                        apps_rest_tracks_right_half += 1
                else:
                    not_none_bboxes_rest_tracks -= 1

        if not_none_bboxes_this_tracks > 0:
            x_avg_pos_this_tracks = round(x_avg_pos_this_tracks/not_none_bboxes_this_tracks, 2)
        else:
            x_avg_pos_this_tracks = None

        if not_none_bboxes_this_tracks > 0:
            y_avg_pos_this_tracks = round(y_avg_pos_this_tracks/not_none_bboxes_this_tracks, 2)
        else:
            y_avg_pos_this_tracks = None

        if not_none_bboxes_rest_tracks > 0:
            x_avg_pos_rest_tracks = round(x_avg_pos_rest_tracks/not_none_bboxes_rest_tracks, 2)
        else:
            x_avg_pos_rest_tracks = None

        if not_none_bboxes_rest_tracks > 0:
            y_avg_pos_rest_tracks = round(y_avg_pos_rest_tracks/not_none_bboxes_rest_tracks, 2)
        else:
            y_avg_pos_rest_tracks = None

        len_this_tracks = len(this_tracks)
        len_rest_tracks = len(rest_tracks)

        this_tracks_per_frame = defaultdict(int)
        for i in range(1,751):
            this_tracks_per_frame[str(i)] = 0
        for track in this_tracks:
            for bbox in track.bboxes:
                bbox_frame = str(bbox.frame)[-3:]
                this_tracks_per_frame[bbox_frame] += 1
        total = sum(list(this_tracks_per_frame.values()))        
        avg_this_tracks_per_frame = round(total/750, 2)

        rest_tracks_per_frame = defaultdict(int)
        for i in range(1,751):
            rest_tracks_per_frame[str(i)] = 0
        for track in rest_tracks:
            for bbox in track.bboxes:
                bbox_frame = str(bbox.frame)[-3:]
                rest_tracks_per_frame[bbox_frame] += 1
        total = sum(list(rest_tracks_per_frame.values()))                
        avg_rest_tracks_per_frame = round(total/750, 2)

        feats = Feats(
                    x_avg_pos_this_tracks=x_avg_pos_this_tracks,
                    y_avg_pos_this_tracks=y_avg_pos_this_tracks,
                    x_avg_pos_rest_tracks=x_avg_pos_rest_tracks,
                    y_avg_pos_rest_tracks=y_avg_pos_rest_tracks,
                    len_this_tracks=len_this_tracks,
                    len_rest_tracks=len_rest_tracks,
                    avg_this_tracks_per_frame=avg_this_tracks_per_frame,
                    avg_rest_tracks_per_frame=avg_rest_tracks_per_frame,
                    apps_this_tracks_left_half=apps_this_tracks_left_half,
                    apps_this_tracks_right_half=apps_this_tracks_right_half,
                    apps_rest_tracks_left_half=apps_rest_tracks_left_half,
                    apps_rest_tracks_right_half=apps_rest_tracks_right_half
                    )  
        if self.neptune_logger is not None:
            chosen_feats = [f.name for f in fields(feats)]
            self.neptune_logger.log(f"feats", str(chosen_feats))
        return feats


if __name__ == "__main__":
    from cluster_classification_dataset import ClusterClassificationDataset

    data_path = "../../data/SoccerNetGS/valid"
    videos_list = ["SNGS-021", "SNGS-039", "SNGS-078"]

    dataset = ClusterClassificationDataset(data_path, videos_list)
    cluster_feats = ClusterFeats()

    X = []
    y = []
    for data, label in dataset:
        feats = cluster_feats.forward(data)
        arr = feats.to_vector()
        X.append(arr)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(X, y)
