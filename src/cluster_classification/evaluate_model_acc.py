from typing import List
import joblib
import numpy as np
from tqdm import tqdm
from functools import reduce
import operator
from dataclasses import dataclass


from real_clusters_dataset import read_track_clustering_file, RealClustersDataset, TrackClustering, TrackCluster
from cluster_classification_dataset import ClusterScenario, get_tracks, Track
from cluster_feats import ClusterFeats
from SNGS_dataset.groundtruth_reader import GroundTruthReader 

@dataclass
class ClassifiedTrack:
    track_id: int
    pred_role_id: int
    

class ClusterPredictionModel():
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.cluster_feats = ClusterFeats()
        
    # Predicts most likely role_id for the cluster, but ruling out a invalid clustering
    def predict_cluster_label(self, cluster_tracks : List[Track], rest_tracks: List[Track]) -> float:
        feats = self.cluster_feats.forward(cluster_tracks, rest_tracks)
        feats_vector = np.array([feats.to_vector()])
        pred_proba = self.model.predict_proba(feats_vector)[0]
        # This method wont return invalid track
        probs_role_ids = pred_proba[:-1]
        highest_prob_class = np.argmax(probs_role_ids)
        return highest_prob_class
    
    # Predicts the probability of this cluster being a valid cluster
    def predict_score_cluster_valid(self, cluster_tracks : List[Track], rest_tracks: List[Track]) -> float:
        feats = self.cluster_feats.forward(cluster_tracks, rest_tracks)
        feats_vector = np.array([feats.to_vector()])
        pred_proba = self.model.predict_proba(feats_vector)
        # The probability of being valid is 1 - prob being invalid
        return 1 - pred_proba[0][6]
    
def predict_cluster_valid_score(cluster: TrackCluster, model, video_tracks: List[Track]):
    cluster_tracks = [t for t in video_tracks if t.track_id in cluster.track_ids]
    rest_tracks = [t for t in video_tracks if t.track_id not in cluster.track_ids]
    
    score_cluster = model.predict_score_cluster_valid(cluster_tracks, rest_tracks)
    
    return score_cluster

def predict_cluster_class(cluster: TrackCluster, model, video_tracks: List[Track]):
    cluster_tracks = [t for t in video_tracks if t.track_id in cluster.track_ids]
    rest_tracks = [t for t in video_tracks if t.track_id not in cluster.track_ids]
    
    pred_cluster_class = model.predict_cluster_label(cluster_tracks, rest_tracks)
    
    return pred_cluster_class
    

def eval_clustering_confidence(clustering: TrackClustering, model, gt_reader: GroundTruthReader) -> float:
    video_tracks: List[Track] = get_tracks([clustering.video], gt_reader)[clustering.video]
    clustering_valid_scores = [predict_cluster_valid_score(cluster, model, video_tracks) for cluster in clustering.clusters]
    # All have to be valid
    clustering_score = reduce(operator.mul, clustering_valid_scores , 1)
    # Mean of being valid
    # clustering_score = np.mean(clustering_valid_scores)
    return clustering_score

def cluster_to_tracks(clustering: TrackClustering, model, gt_reader: GroundTruthReader) -> List[ClassifiedTrack]:
    video_tracks: List[Track] = get_tracks([clustering.video], gt_reader)[clustering.video]
    classified_tracks: List[ClassifiedTrack] = []
    for cluster in clustering.clusters:
        cluster_class = predict_cluster_class(cluster, model, video_tracks)
        for track_id in cluster.track_ids:
            classified_tracks.append(ClassifiedTrack(
                track_id=track_id,
                pred_role_id=cluster_class
            ))
    return classified_tracks
    
def evaluate_track_accuracy(pred_tracks: List[ClassifiedTrack], gt_tracks: List[Track] ) -> float:
     # Create a map from track_id to role_id for ground truth tracks
    gt_track_map = {track.track_id: track for track in gt_tracks}

    correct_tracks = 0
    correct_bboxes = 0

    # Evaluate each predicted track
    for pred_track in pred_tracks:
        if pred_track.track_id in gt_track_map and pred_track.pred_role_id == gt_track_map[pred_track.track_id].role_id:
            correct_tracks += 1
            correct_bboxes += len(gt_track_map[pred_track.track_id].bboxes)

    # Calculate track accuracy
    track_accuracy = correct_tracks / len(pred_tracks) if pred_tracks else 0.0
    
    # Calculate bbox accuracy
    total_bboxes = sum([len(track.bboxes) for track in gt_tracks])
    bbox_accuracy = correct_bboxes / total_bboxes if total_bboxes else 0.0
    
    return track_accuracy, bbox_accuracy

def evaluate_clustering_on_real_data(data_path: str, real_cluster_path: str, model_path: str):
    # Load Data
    gt_reader: GroundTruthReader = GroundTruthReader(data_path)
    real_dataset: List[TrackClustering] = read_track_clustering_file(real_cluster_path)
    videos_list = np.unique([c.video for c in real_dataset])
    
    # Load Model
    model = ClusterPredictionModel(model_path)
    
    # Evaluate each video
    videos_correct_clustering = []
    videos_track_acc, videos_bbox_acc = [], []
    best_videos_track_acc, best_videos_bbox_acc= [], []
    for video in tqdm(videos_list, total=len(videos_list), desc='Iterating over videos'):
        # Evaluate clusterings for every possible kmeans
        video_clusterings: List[TrackClustering] = [c for c in real_dataset if c.video == video]
        clustering_confidences = [eval_clustering_confidence(c, model, gt_reader) for c in video_clusterings]
            
        # The best clustering is that one with greatest valid probability
        best_clustering: TrackClustering = video_clusterings[clustering_confidences.index(max(clustering_confidences))]
        true_clustering = next((c for c in video_clusterings if c.is_valid))
        
        # Was chosen clustering the correct one?
        is_video_clustering_correct = best_clustering.is_valid
        if is_video_clustering_correct:
            videos_correct_clustering.append(video)
        
        # Get Metrics per-track
        track_predictions = cluster_to_tracks(best_clustering, model, gt_reader)
        video_tracks: List[Track] = get_tracks([video], gt_reader)[video]
        track_acc, bbox_acc = evaluate_track_accuracy(track_predictions, video_tracks)
        videos_track_acc.append(track_acc)
        videos_bbox_acc.append(bbox_acc)
        
        # Get Metrics for right clustering
        best_track_predictions = cluster_to_tracks(true_clustering, model, gt_reader)
        best_track_acc, best_bbox_acc = evaluate_track_accuracy(best_track_predictions, video_tracks)
        best_videos_track_acc.append(best_track_acc)
        best_videos_bbox_acc.append(best_bbox_acc)
        
#         print("/////")
#         print(true_clustering)
#         for cluster in true_clustering.clusters:
#             print(cluster)
#             gt_roles = [next((v for v in video_tracks if v.track_id == t)).role_id for t in cluster.track_ids]
#             print(gt_roles)
#         print(best_track_predictions)
        
     
    correct_clustering = len(videos_correct_clustering) / len(videos_list) * 100
    print(f"% of videos with correct clustering found: {correct_clustering:.2f}%")
    
    avg_track_acc = np.mean(videos_track_acc) * 100
    print(f"Average Track role_id Accuracy {avg_track_acc:.2f}")
    
    avg_bbox_acc = np.mean(videos_bbox_acc) * 100
    print(f"Average Bbox role_id Accuracy {avg_bbox_acc:.2f}")
    
    best_avg_track_acc = np.mean(best_videos_track_acc) * 100
    print(f"True Clustering Track role_id Accuracy {best_avg_track_acc:.2f}")
    
    best_avg_bbox_acc = np.mean(best_videos_bbox_acc) * 100
    print(f"True Clustering Average Bbox role_id Accuracy {best_avg_bbox_acc:.2f}")
      
             
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classification model.")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="../../data/SoccerNetGS/valid",
        help="Path to the dataset.",
    )    
    parser.add_argument(
        "--real_cluster_path",
        type=str,
        required=False,
        default=None,
        help="Path to a json with RealClusters Dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="Path to a saved model",
    )
    args = vars(parser.parse_args())
    
    model = args["model"]
    data_path = args["data_path"]
    real_cluster_path = args["real_cluster_path"]
    
    evaluate_clustering_on_real_data(data_path, real_cluster_path, model)