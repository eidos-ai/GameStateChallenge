import numpy as np
from dataclasses import dataclass, astuple
from collections import defaultdict
from typing import List, Optional

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
    bboxes: List[BBox]
    def __str__(self):
        return f'Track(track_id={self.track_id}, bboxes=(lenght: {len(self.bboxes)}))'
    def __repr__(self):
        return self.__str__()


@dataclass
class ClusterScenario:
    this_tracks_role_id: Optional[int]
    this_tracks: List[Track]
    rest_tracks: List[Track]
    is_valid: bool
    def __str__(self):
        return f'ClusterScenario(this_tracks_role_id={self.this_tracks_role_id}, this_tracks=(lenght: {len(self.this_tracks)}), rest_tracks=(lenght: {len(self.rest_tracks)}), is_valid={self.is_valid})'
    def __repr__(self):
        return self.__str__()


@dataclass
class Feats:
    x_avg_pos_this_tracks: float | None
    y_avg_pos_this_tracks: float | None
    x_avg_pos_rest_tracks: float | None
    y_avg_pos_rest_tracks: float | None
    
    bbox_per_frame_this_tracks: int
    bbox_per_frame_rest_tracks: int
    
    apps_this_tracks_left_half: int
    apps_this_tracks_right_half: int
    apps_rest_tracks_left_half: int
    apps_rest_tracks_right_half: int

    def __str__(self):
        return f'Feats(\n    x_avg_pos_this_tracks={self.x_avg_pos_this_tracks}, \n    y_avg_pos_this_tracks={self.y_avg_pos_this_tracks}, \n    x_avg_pos_rest_tracks={self.x_avg_pos_rest_tracks}, \n    y_avg_pos_rest_tracks={self.y_avg_pos_rest_tracks}, \n    bbox_per_frame_this_tracks={self.bbox_per_frame_this_tracks}, \n    bbox_per_frame_rest_tracks={self.bbox_per_frame_rest_tracks}, \n  apps_this_tracks_left_half={self.apps_this_tracks_left_half}, \n    apps_this_tracks_right_half={self.apps_this_tracks_right_half}, \n    apps_rest_tracks_left_half={self.apps_rest_tracks_left_half}, \n    apps_rest_tracks_right_half={self.apps_rest_tracks_right_half} \n)'
    def __repr__(self):
        return self.__str__()

    def to_vector(self):
        return np.array(astuple(self))


def average_bboxes_per_frame(tracks: List[Track]):
    frame_counts = defaultdict(int)

    # Count the number of bboxes for each frame across all tracks
    for track in tracks:
        for bbox in track.bboxes:
            frame_counts[bbox.frame] += 1

    # Compute the average number of bboxes per frame
    total_bboxes = sum(frame_counts.values())
    num_frames = len(frame_counts)
    average_bboxes = total_bboxes / num_frames if num_frames > 0 else 0

    return average_bboxes

class ClusterFeats():
    def __init__(self):
        pass
        
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
    
        bbox_per_frame_this_tracks = average_bboxes_per_frame(this_tracks)
        bbox_per_frame_rest_tracks = average_bboxes_per_frame(rest_tracks)

        feats = Feats(
                    x_avg_pos_this_tracks=x_avg_pos_this_tracks,
                    y_avg_pos_this_tracks=y_avg_pos_this_tracks,
                    x_avg_pos_rest_tracks=x_avg_pos_rest_tracks,
                    y_avg_pos_rest_tracks=y_avg_pos_rest_tracks,
            
                    bbox_per_frame_this_tracks=bbox_per_frame_this_tracks,
                    bbox_per_frame_rest_tracks=bbox_per_frame_rest_tracks,
            
                    apps_this_tracks_left_half=apps_this_tracks_left_half,
                    apps_this_tracks_right_half=apps_this_tracks_right_half,
                    apps_rest_tracks_left_half=apps_rest_tracks_left_half,
                    apps_rest_tracks_right_half=apps_rest_tracks_right_half
                    )
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
