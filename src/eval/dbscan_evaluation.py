import os
import sys
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from SNGS_dataset.groundtruth_reader import GroundTruthReader
from dataclasses import asdict, dataclass
from collections import defaultdict
from typing import Any, List, Dict
from dbscan_clustering import DBScanClustering
from tqdm import tqdm
from sklearn import metrics


@dataclass(frozen=True)
class Metrics:
    video: str
    predicted_clusters: int
    homogeneity: float
    completeness: float


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


def load_data(
    dataset_path: str,
    embeddings_directory: str,
    videos_list: List[str],
) -> Any:
    """Loads data for evaluation"""
    if videos_list is None:
        videos_list = [
            video for video in os.listdir(dataset_path) if video[:5] != "SNGS-"
        ]
    gt_reader = GroundTruthReader(dataset_path, videos_list)
    embeddings_dict = defaultdict(lambda: defaultdict(list))
    groundtruth_dict = defaultdict(lambda: defaultdict(list))

    for video in tqdm(videos_list, total=len(videos_list)):
        # Get embeddings and annotations for this video in correct order
        video_groundtruth = []
        video_gt = gt_reader.get_annotations_from_video(video)
        video_annotations_ids, video_embeddings = load_embeddings(
            embeddings_directory, video
        )
        assert len(video_annotations_ids) == len(
            video_embeddings
        ), "The annotations and embeddings lists have different lengths"
        for ann_id in video_annotations_ids:
            for annotation in video_gt:
                if annotation.annotation_id == ann_id:
                    video_groundtruth.append(annotation.role_id)
                    break
        # At this point, I have video_groundtruth and video_embeddings
        embeddings_dict[video] = video_embeddings
        groundtruth_dict[video] = video_groundtruth
    return embeddings_dict, groundtruth_dict


def get_average_metrics(metrics_list: List[Metrics]):
    """Get the average metrics of all videos."""
    avg_dict = defaultdict(float)
    for video_metrics in metrics_list:
        temp_dict = asdict(video_metrics)
        for metric in temp_dict.keys():
            if metric != "predicted_clusters" and metric != "video":
                avg_dict[f"{metric}_average"] += temp_dict[metric]
    for metric in avg_dict.keys():
        avg_dict[metric] /= len(metrics_list)
        avg_dict[metric] = round(avg_dict[metric], 3)
    return avg_dict


class DBScanEvaluation:
    def videos_dbscan_evaluation(
        self,
        annotations_dict: Dict[str, Any],
        embeddings_dict: Dict[str, Any],
        clustering_method: DBScanClustering,
    ) -> Any:
        """Evaluates clustering for all videos"""
        assert len(annotations_dict.keys()) == len(
            embeddings_dict.keys()
        ), "Different number of videos in annotations and embeddings"
        total_metrics = []
        for video in annotations_dict.keys():
            predictions = clustering_method.fit(embeddings_dict[video])
            clusters_count = len(set(predictions))
            video_metrics = Metrics(
                video=video,
                predicted_clusters=(
                    clusters_count - 1 if -1 in predictions else clusters_count
                ),
                homogeneity=round(
                    metrics.homogeneity_score(annotations_dict[video], predictions), 3
                ),
                completeness=round(
                    metrics.completeness_score(annotations_dict[video], predictions), 3
                ),
            )
            total_metrics.append(video_metrics)
        average_metrics = get_average_metrics(total_metrics)

        return average_metrics, total_metrics


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

    clustering_evaluation = DBScanEvaluation()
    clustering_method = DBScanClustering(eps=eps, min_samples=min_samples)
    embeddings_dict, annotations_dict = load_data(
        data_path, embeddings_directory, videos_list
    )
    average_dict, metrics_list = clustering_evaluation.videos_dbscan_evaluation(
        annotations_dict, embeddings_dict, clustering_method
    )

    for item in metrics_list:
        print(item)

    print(json.dumps(average_dict, indent=4, cls=MyEncoder))
