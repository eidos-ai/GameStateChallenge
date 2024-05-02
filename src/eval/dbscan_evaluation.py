import os
import sys
import torch
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from SNGS_dataset.groundtruth_reader import GroundTruthReader
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from dataclasses import asdict, dataclass, fields
from collections import defaultdict
from typing import Any
from dbscan_clustering import DBScanClustering
from tqdm import tqdm
from sklearn import metrics


def load_embeddings(embeddings_directory: str, video_name: str) -> Any:
    """Load embeddings from a given video"""
    embeddings_file_path = os.path.join(
        embeddings_directory, f"{video_name}_embeddings.npz"
    )
    with np.load(embeddings_file_path) as data:
        try:
            model_path = data["model_path"][0]
        except:
            pass
        annotation_ids = data["annotation_ids"]
        embeddings = data["embeddings"]
    return annotation_ids, embeddings


def get_average_dict(metrics_dict):
    """Get the average metrics of all videos."""
    avg_dict = defaultdict(float)
    for video in metrics_dict.keys():
        for metric in metrics_dict[video].keys():
            if metric != "predicted_clusters":
                avg_dict[f"{metric}_average"] += metrics_dict[video][metric]
    for metric in avg_dict.keys():
        avg_dict[metric] /= len(metrics_dict)
        avg_dict[metric] = round(avg_dict[metric], 3)
    return avg_dict


@dataclass(frozen=True)
class Metrics:
    predicted_clusters: int
    homogeneity: float
    completeness: float


class DBScanEvaluation:
    def __init__(self, videos_list: list[str] = None):
        self.videos_list = videos_list

    def video_dbscan_evaluation(
        self,
        gt_cluster_ids: list[int],
        embeddings: list[Any],
        clustering_method: DBScanClustering,
        video: str,
    ) -> dict[str:Any]:
        """Evaluates clustering from on-memory data"""
        assert len(gt_cluster_ids) == len(
            embeddings
        ), "The annotations and embeddings lists have different lengths"
        predictions = clustering_method.fit(embeddings)
        clusters_count = len(set(predictions))
        video_metrics = Metrics(
            predicted_clusters=(
                clusters_count - 1 if -1 in predictions else clusters_count
            ),
            homogeneity=round(
                metrics.homogeneity_score(gt_cluster_ids, predictions), 3
            ),
            completeness=round(
                metrics.completeness_score(gt_cluster_ids, predictions), 3
            ),
        )
        # metrics_dict = defaultdict(float)
        # metrics_dict["clusters_count"] = clusters_count - 1 if -1 in clusters_count else clusters_count
        # metrics_dict["homogeneity"] = round(metrics.homogeneity_score(gt_cluster_ids, predictions), 3)
        # metrics_dict["completeness"] = round(metrics.completeness_score(gt_cluster_ids, predictions), 3)
        # metrics_dict["adjusted_rand_index"] = round(metrics.adjusted_rand_score(gt_cluster_ids, predictions), 3)
        # metrics_dict["silhouette_coefficient"] = round(metrics.silhouette_score(embeddings, predictions), 3)
        # return metrics_dict
        return video_metrics

    def stored_videos_dbscan_evaluation(
        self,
        dataset_path: str,
        embeddings_directory: str,
        clustering_method: DBScanClustering,
    ) -> dict[str:Any]:
        """Loads data to evaluate clustering"""
        if self.videos_list is None:
            self.videos_list = [
                video for video in os.listdir(dataset_path) if video[:5] != "SNGS-"
            ]
        gt_reader = GroundTruthReader(dataset_path, self.videos_list)
        metrics_dict = defaultdict(lambda: defaultdict(list))
        for video in tqdm(self.videos_list, total=len(self.videos_list)):
            # Get embeddings and annotations for this video in correct order
            groundtruth = []
            video_gt = gt_reader.get_annotations_from_video(video)
            video_embeddings_annotations_ids, video_embeddings = load_embeddings(
                embeddings_directory, video
            )
            for ann_id in video_embeddings_annotations_ids:
                for annotation in video_gt:
                    if annotation.annotation_id == ann_id:
                        groundtruth.append(annotation.role_id)
                        break
            # Perform clustering fit and eval with video embeddings and annotations
            video_metrics = self.video_dbscan_evaluation(
                groundtruth, video_embeddings, clustering_method, video
            )
            metrics_dict[video] = asdict(video_metrics)
        average_dict = get_average_dict(metrics_dict)
        return average_dict, metrics_dict


if __name__ == "__main__":
    import argparse
    import json

    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.float32):
                return float(obj)
            return super(MyEncoder, self).default(obj)

    parser = argparse.ArgumentParser(description="Script for clustering evaluation.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,  # example: ../../data/SoccerNetGS/train
        help="Train, valid or test folder path.",
    )
    parser.add_argument(
        "--videos_list",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="List of videos to use for clustering evaluation",
    )
    parser.add_argument(
        "--embeddings_directory",
        type=str,
        required=False,
        default="./video_files",
        help="Directory where the embeddings (npz files) are.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        required=False,
        default=0.5,
        help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        required=False,
        default=5,
        help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.",
    )

    args = vars(parser.parse_args())
    data_path = args["data_path"]
    videos_list = args["videos_list"]
    embeddings_directory = args["embeddings_directory"]
    eps = args["eps"]
    min_samples = args["min_samples"]

    clustering_evaluation = DBScanEvaluation(videos_list)
    clustering_method = DBScanClustering(eps=eps, min_samples=min_samples)
    average_dict, metrics_dict = clustering_evaluation.stored_videos_dbscan_evaluation(
        data_path, embeddings_directory, clustering_method
    )

    print(json.dumps(metrics_dict, indent=4, cls=MyEncoder))
    print(json.dumps(average_dict, indent=4, cls=MyEncoder))
