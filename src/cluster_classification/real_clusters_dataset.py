from dataclasses import dataclass
from typing import List, Optional
import json
from torch.utils.data import Dataset
from cluster_classification_dataset import ClusterScenario, get_tracks
from SNGS_dataset.groundtruth_reader import GroundTruthReader

@dataclass
class BboxCluster:
    cluster_id: str
    annotation_ids: List[int]

@dataclass
class BBoxClustering:
    clusters: List[BboxCluster]
    video: str
    is_valid: bool

@dataclass
class TrackCluster:
    cluster_id: str
    track_ids: List[int]
    gt_role_id: Optional[int]

@dataclass
class TrackClustering:
    clusters: List[TrackCluster]
    video: str
    is_valid: bool
    

class RealClustersDataset(Dataset):
    def __init__(self, data_path: str, cluster_path: str, videos_list: List[str]=None) -> None:
        clusterings: List[TrackClustering] = read_track_clustering_file(cluster_path)
        gt_reader = GroundTruthReader(data_path, videos_list)
        all_tracks = get_tracks(videos_list, gt_reader)
        self.cluster_scenarios = []
        for clustering in clusterings:
            for c in clustering.clusters:
                this_tracks = [t for t in all_tracks[clustering.video] if t.track_id in c.track_ids]
                other_tracks = [t for t in all_tracks[clustering.video] if t.track_id not in c.track_ids]
                cluster_scenario = ClusterScenario(                    
                    this_tracks_role_id=c.gt_role_id,
                    this_tracks=this_tracks,
                    rest_tracks=other_tracks,
                    is_valid=clustering.is_valid
                )
                self.cluster_scenarios.append(cluster_scenario)
        print("finished reading")
        
    def __len__(self) -> int:
        return len(self.cluster_scenarios)

    def __getitem__(self, idx: int) -> ClusterScenario:
        return self.cluster_scenarios[idx]

def read_track_clustering_file(clusters_path: str) -> List[TrackClustering]:
    with open(clusters_path, 'r') as file:
        track_clusters_obj = json.load(file)
        track_clusters = [dict_to_track_clustering(t) for t in track_clusters_obj ]
        
    return track_clusters

def save_track_clustering_file(clusters: List[TrackClustering], out_path: str) -> None:    
    track_clusters_obj = []
    for c in clusters:
        track_clusters_obj.append(track_clustering_to_dict(c))        
    with open(out_path, 'w') as file:
        file.write(json.dumps(track_clusters_obj, indent=4))
        
def track_clustering_to_dict(track_clustering: TrackClustering) -> dict:
    def track_clustering_to_dict(tc: TrackClustering) -> dict:
        return {
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "track_ids": cluster.track_ids,
                    "gt_role_id": cluster.gt_role_id
                } for cluster in tc.clusters
            ],
            "video": tc.video,
            "is_valid": tc.is_valid
        }
    
    track_clustering_dict = track_clustering_to_dict(track_clustering)
    
    return track_clustering_dict

def dict_to_track_clustering(data: dict) -> TrackClustering:
   # Parse the 'clusters' part of the dictionary
    clusters = [
        TrackCluster(
            cluster_id=cluster_dict["cluster_id"],
            track_ids=cluster_dict["track_ids"],
            gt_role_id=cluster_dict["gt_role_id"]
        ) for cluster_dict in data["clusters"]
    ]
    
    # Extract the 'video' and 'is_valid' fields
    video = data["video"]
    is_valid = data["is_valid"]
    
    # Create the TrackClustering dataclass instance
    return TrackClustering(clusters=clusters, video=video, is_valid=is_valid)


        
if __name__ == "__main__":
    import argparse
    data = RealClustersDataset("../../data/SoccerNetGS/valid", "testdata.json")
    print("RealClustersDataset length", len(data))
    for d in data:
        pass