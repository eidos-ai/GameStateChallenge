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


@dataclass
class ClassifiedTrack:
    track_id: int
    pred_role_id: int
    

# Runs the whole cluster classification pipeline
class ClusterPredictionModel():
    def __init__(self, prediction_model):
        self.prediction_model = prediction_model
        self.cluster_feats = ClusterFeats()
        
    # Predicts most likely role_id for the cluster, but ruling out a invalid clustering
    def predict_cluster_label(self, cluster_tracks : List[Track], rest_tracks: List[Track]) -> float:
        feats = self.cluster_feats.forward(cluster_tracks, rest_tracks)
        feats_vector = np.array([feats.to_vector()])
        pred_proba = self.prediction_model.predict_proba(feats_vector)[0]
        # This method wont return invalid track
        probs_role_ids = pred_proba[:-1]
        highest_prob_class = np.argmax(probs_role_ids)
        return highest_prob_class
    
    # Predicts the probability of this cluster being a valid cluster
    def predict_score_cluster_valid(self, cluster_tracks : List[Track], rest_tracks: List[Track]) -> float:
        feats = self.cluster_feats.forward(cluster_tracks, rest_tracks)
        feats_vector = np.array([feats.to_vector()])
        pred_proba = self.prediction_model.predict_proba(feats_vector)
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
    

def eval_clustering_confidence(clustering: TrackClustering, model) -> float:
    video_tracks: List[Track] = clustering.gt_tracks
    clustering_valid_scores = [predict_cluster_valid_score(cluster, model, video_tracks) for cluster in clustering.clusters]
    
    # All have to be valid
    # clustering_score = reduce(operator.mul, clustering_valid_scores , 1)
    # neptune_logger.log(f"evaluation/clustering_score_method", "product")
    
    # Mean of being valid
    # clustering_score = np.mean(clustering_valid_scores)
    # neptune_logger.log(f"evaluation/clustering_score_method", "mean")
    
    # Weighted mean of being valid
    clustering_valid_scores = [len(cluster.track_ids) * predict_cluster_valid_score(cluster, model, video_tracks) for cluster in clustering.clusters]
    total_cluster_size = sum(len(cluster.track_ids) for cluster in clustering.clusters)
    clustering_score = np.sum(clustering_valid_scores) / total_cluster_size
    # neptune_logger.log(f"evaluation/clustering_score_method", "weigh")
    return clustering_score

def cluster_to_tracks(clustering: TrackClustering, model) -> List[ClassifiedTrack]:
    video_tracks: List[Track] = clustering.gt_tracks
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

@dataclass
class VideoMetrics:
    video: str
    track_accuracy: float
    bbox_accuracy: float
    best_track_accuracy: float
    best_bbox_accuracy: float
    is_clustering_correct: bool
    
    def __str__(self):
        return f'VideoMetrics(\n    video={self.video}, \n    track_accuracy={self.track_accuracy}, \n    bbox_accuracy={self.bbox_accuracy}, \n    best_track_accuracy={self.best_track_accuracy}, \n    best_bbox_accuracy={self.best_bbox_accuracy}, \n    is_clustering_correct={self.is_clustering_correct} \n)'
    def __repr__(self):
        return self.__str__()

@dataclass
class ClusteringEvaluationMetrics:
    average_track_accuracy: float
    average_bbox_accuracy: float
    average_best_track_accuracy: float
    average_best_bbox_accuracy: float
    percent_correct_clustering: float
    video_metrics: List[VideoMetrics]

# Given a ClusteringDataset, runs a model to find which is the best clustering on each scenario, and classifies clusters
# Then compares found role_ids vs ground trutth annotations of tracks
def evaluate_clustering(clustering_dataset: RealClustersDataset, prediction_model) -> ClusteringEvaluationMetrics:
    model = ClusterPredictionModel(prediction_model)
    
    clustering: List[TrackClustering] = clustering_dataset.get_clusterings()
    videos_list = np.unique([c.video for c in clustering])
    
    video_metrics_list = []
    videos_correct_clustering = []
    videos_track_acc, videos_bbox_acc = [], []
    best_videos_track_acc, best_videos_bbox_acc = [], []
    
    for video in tqdm(videos_list, total=len(videos_list), desc='Iterating over videos'):
        video_clusterings: List[TrackClustering] = [c for c in clustering if c.video == video]
        clustering_confidences = [eval_clustering_confidence(c, model) for c in video_clusterings]
        
        best_clustering: TrackClustering = video_clusterings[clustering_confidences.index(max(clustering_confidences))]
        true_clustering = next((c for c in video_clusterings if c.is_valid))
        
        is_video_clustering_correct = best_clustering.is_valid
        if is_video_clustering_correct:
            videos_correct_clustering.append(video)
        
        track_predictions = cluster_to_tracks(best_clustering, model)
        video_tracks: List[Track] = best_clustering.gt_tracks
        track_acc, bbox_acc = evaluate_track_accuracy(track_predictions, video_tracks)
        videos_track_acc.append(track_acc)
        videos_bbox_acc.append(bbox_acc)
        
        best_track_predictions = cluster_to_tracks(true_clustering, model)
        best_track_acc, best_bbox_acc = evaluate_track_accuracy(best_track_predictions, video_tracks)
        best_videos_track_acc.append(best_track_acc)
        best_videos_bbox_acc.append(best_bbox_acc)
        
        video_metrics_list.append(VideoMetrics(
            video=video,
            track_accuracy=track_acc,
            bbox_accuracy=bbox_acc,
            best_track_accuracy=best_track_acc,
            best_bbox_accuracy=best_bbox_acc,
            is_clustering_correct=is_video_clustering_correct
        ))
    
    average_track_accuracy = np.mean(videos_track_acc)
    average_bbox_accuracy = np.mean(videos_bbox_acc)
    average_best_track_accuracy = np.mean(best_videos_track_acc)
    average_best_bbox_accuracy = np.mean(best_videos_bbox_acc)
    percent_correct_clustering = len(videos_correct_clustering) / len(videos_list) * 100
    
    return ClusteringEvaluationMetrics(
        average_track_accuracy=average_track_accuracy,
        average_bbox_accuracy=average_bbox_accuracy,
        average_best_track_accuracy=average_best_track_accuracy,
        average_best_bbox_accuracy=average_best_bbox_accuracy,
        percent_correct_clustering=percent_correct_clustering,
        video_metrics=video_metrics_list
    )
    
    
def evaluate_model_on_clustering(data_path: str, real_cluster_path: str, model_path: str):
    # Load Data
    clustering_dataset = RealClustersDataset(data_path, real_cluster_path)

    # Load Model
    prediction_model = joblib.load(model_path)
    
    metrics = evaluate_clustering(clustering_dataset, prediction_model)
     
    print(f"% of videos with correct clustering found: {metrics.percent_correct_clustering:.2f}%")

    print(f"Average Track role_id Accuracy {metrics.average_track_accuracy * 100:.2f}")

    print(f"Average Bbox role_id Accuracy {metrics.average_bbox_accuracy * 100:.2f}")

    print(f"True Clustering Track role_id Accuracy {metrics.average_best_track_accuracy * 100:.2f}")

    print(f"True Clustering Average Bbox role_id Accuracy {metrics.average_best_bbox_accuracy * 100:.2f}")
             
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
    
    model_path = args["model"]
    data_path = args["data_path"]
    real_cluster_path = args["real_cluster_path"]
    
   
    evaluate_model_on_clustering(data_path, real_cluster_path, model_path)