from dataclasses import dataclass
from cluster_classification_dataset import ClusterScenario

@dataclass
class Feats:
    x_avg_pos_this_tracks: float
    y_avg_pos_this_tracks: float
    x_avg_pos_rest_tracks: float
    y_avg_pos_rest_tracks: float
    len_this_tracks: int
    len_rest_tracks: int
    avg_this_tracks_per_frame: float
    avg_rest_tracks_per_frame: float


class ClusterFeats():
    def forward(self, cluster_scenario: ClusterScenario) -> Feats:
        x_avg_pos_this_tracks = 0
        y_avg_pos_this_tracks = 0
        x_avg_pos_rest_tracks = 0
        y_avg_pos_rest_tracks = 0

        len_this_tracks = len(cluster_scenario.this_tracks)
        len_rest_tracks = len(cluster_scenario.rest_tracks)

        for track in cluster_scenario.this_tracks:
            for bbox in track.bboxes:
                try:
                    x_avg_pos_this_tracks += bbox.x_pitch
                    y_avg_pos_this_tracks += bbox.y_pitch
                except:
                    len_this_tracks -= 1

        for track in cluster_scenario.rest_tracks:
            for bbox in track.bboxes:
                try:
                    x_avg_pos_rest_tracks += bbox.x_pitch
                    y_avg_pos_rest_tracks += bbox.y_pitch
                except:
                    len_rest_tracks -= 1

        x_avg_pos_this_tracks /= len_this_tracks
        y_avg_pos_this_tracks /= len_this_tracks
        x_avg_pos_rest_tracks /= len_rest_tracks
        y_avg_pos_rest_tracks /= len_rest_tracks

        avg_this_tracks_per_frame = 0
        avg_rest_tracks_per_frame = 0
        
        len_this_tracks = len(cluster_scenario.this_tracks)
        len_rest_tracks = len(cluster_scenario.rest_tracks)
        
        feats = Feats(x_avg_pos_this_tracks=x_avg_pos_this_tracks,
                            y_avg_pos_this_tracks=y_avg_pos_this_tracks,
                            x_avg_pos_rest_tracks=x_avg_pos_rest_tracks,
                            y_avg_pos_rest_tracks=y_avg_pos_rest_tracks,
                            len_this_tracks=len_this_tracks,
                            len_rest_tracks=len_rest_tracks,
                            avg_this_tracks_per_frame=avg_this_tracks_per_frame,
                            avg_rest_tracks_per_frame=avg_rest_tracks_per_frame
                           )
        return feats


if __name__ == "__main__":
    from cluster_classification_dataset import ClusterClassificationDataset

    data_path = "../../data/SoccerNetGS/valid"
    videos_list = ["SNGS-021", "SNGS-039", "SNGS-078"]

    cluster_feats = ClusterFeats()
    dataset = ClusterClassificationDataset(data_path, videos_list)

    total_feats = []
    for i in range(len(dataset)):
        feats = cluster_feats.forward(dataset[i])
