import numpy as np
from dataclasses import dataclass, astuple
from collections import defaultdict
from cluster_classification_dataset import ClusterScenario

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

    def __str__(self):
        return f'Feats(\n    x_avg_pos_this_tracks={self.x_avg_pos_this_tracks}, \n    y_avg_pos_this_tracks={self.y_avg_pos_this_tracks}, \n    x_avg_pos_rest_tracks={self.x_avg_pos_rest_tracks}, \n    y_avg_pos_rest_tracks={self.y_avg_pos_rest_tracks}, \n    len_this_tracks={self.len_this_tracks}, \n    len_rest_tracks={self.len_rest_tracks}, \n    avg_this_tracks_per_frame={self.avg_this_tracks_per_frame}, \n    avg_rest_tracks_per_frame={self.avg_rest_tracks_per_frame}\n)'
    def __repr__(self):
        return self.__str__()


class ClusterFeats():
    def forward(self, cluster_scenario: ClusterScenario) -> Feats:
        x_avg_pos_this_tracks = 0
        y_avg_pos_this_tracks = 0
        x_avg_pos_rest_tracks = 0
        y_avg_pos_rest_tracks = 0

        len_this_tracks = len(cluster_scenario.this_tracks)
        len_rest_tracks = len(cluster_scenario.rest_tracks)

        not_none_bboxes_this_tracks = 0
        not_none_bboxes_rest_tracks = 0

        for track in cluster_scenario.this_tracks:
            not_none_bboxes_this_tracks += len(track.bboxes)

            for bbox in track.bboxes:
                if bbox.x_pitch is not None or bbox.y_pitch is not None:
                    x_avg_pos_this_tracks += bbox.x_pitch
                    y_avg_pos_this_tracks += bbox.y_pitch
                else:
                    not_none_bboxes_this_tracks -= 1

        for track in cluster_scenario.rest_tracks:
            not_none_bboxes_rest_tracks += len(track.bboxes)

            for bbox in track.bboxes:
                if bbox.x_pitch is not None or bbox.y_pitch is not None:
                    x_avg_pos_rest_tracks += bbox.x_pitch
                    y_avg_pos_rest_tracks += bbox.y_pitch
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

        len_this_tracks = len(cluster_scenario.this_tracks)
        len_rest_tracks = len(cluster_scenario.rest_tracks)

        this_tracks_per_frame = defaultdict(int)
        for i in range(1,751):
            this_tracks_per_frame[str(i)] = 0
        for track in cluster_scenario.this_tracks:
            for bbox in track.bboxes:
                bbox_frame = str(bbox.frame)[-3:]
                this_tracks_per_frame[bbox_frame] += 1
        total = sum(list(this_tracks_per_frame.values()))        
        avg_this_tracks_per_frame = round(total/750, 2)

        rest_tracks_per_frame = defaultdict(int)
        for i in range(1,751):
            rest_tracks_per_frame[str(i)] = 0
        for track in cluster_scenario.rest_tracks:
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
                    avg_rest_tracks_per_frame=avg_rest_tracks_per_frame
                    )
        return feats

    def feats_to_vector(self, feats: Feats) -> np.array:
        return np.array(astuple(feats))


if __name__ == "__main__":
    from cluster_classification_dataset import ClusterClassificationDataset

    data_path = "../../data/SoccerNetGS/valid"
    videos_list = ["SNGS-021", "SNGS-039", "SNGS-078"]

    cluster_feats = ClusterFeats()
    dataset = ClusterClassificationDataset(data_path, videos_list)

    total_feats = []
    for i in range(len(dataset)):
        feats = cluster_feats.forward(dataset[i])
        arr = cluster_feats.feats_to_vector(feats)
        print(arr)
