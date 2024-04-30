from dataclasses import asdict
import os
import sys
import torch
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from SNGS_dataset.groundtruth_reader import GroundTruthReader
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from typing import Any, List
from magic_kmeans_clustering import MagicKmeansClustering
from dbscan_clustering import DBScanClustering
from tqdm import tqdm

def load_embeddings(embeddings_directory: str, video_name: str) -> Any:
        '''Load embeddings from a given video'''
        embeddings_file_path = os.path.join(embeddings_directory, f"{video_name}_embeddings.npz")
        with np.load(embeddings_file_path) as data:
            try:
                model_path = data['model_path'][0]
            except:
                pass
            annotation_ids = data['annotation_ids']
            embeddings = data['embeddings']
        return annotation_ids, embeddings

def clustering_accuracy(labels_true: List[int], labels_pred: List[int]):
    # Create the confusion matrix
    cm = confusion_matrix(labels_true, labels_pred, labels=[0, 1, 2, 3, 4])

    # Find the optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)

    # Calculate the overall accuracy using the optimal assignment
    overall_accuracy = cm[row_ind, col_ind].sum() / cm.sum()

    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i, (row, col) in enumerate(zip(row_ind, col_ind)):
        row_sum = np.sum(cm[row, :])
        if row_sum == 0:
            class_accuracy = None
        else:
            class_accuracy = cm[row, col] / row_sum
        per_class_accuracy[int(row)] = class_accuracy

    return overall_accuracy, per_class_accuracy

def get_accuracy_average(accuracy_dict):
    '''Get the average accuracy of all videos.'''
    overall_acc = 0
    per_class_nones = defaultdict(int)
    per_class_acc = defaultdict(float)
    for video in accuracy_dict.keys():
        overall_acc += accuracy_dict[video]["overall"]
        for class_idx in accuracy_dict[video]["per_class"].keys():
            if accuracy_dict[video]["per_class"][class_idx] is None:
                per_class_nones[class_idx] += 1
                continue
            per_class_acc[int(class_idx)] += accuracy_dict[video]["per_class"][class_idx]
    overall_acc /= len(accuracy_dict)
    for class_idx in per_class_acc.keys():
        per_class_acc[class_idx] /= (len(accuracy_dict) - per_class_nones[class_idx])
    avg_dict = defaultdict(lambda: defaultdict(list))
    avg_dict["overall"] = overall_acc
    avg_dict["per_class"] = per_class_acc
    return avg_dict


class ClusteringAccuracyEvaluation():
    def __init__(self, videos_list: list[str]=None):
        self.videos_list = videos_list

    def video_clustering_accuracy(self, gt_cluster_ids: list[int], embeddings: list[Any], 
                                   clustering_method: MagicKmeansClustering, video: str) -> dict[str: Any]:
        '''Evaluates clustering from on-memory data'''
        assert len(gt_cluster_ids) == len(embeddings), "The annotations and embeddings lists have different lengths"
        predictions = clustering_method.fit(embeddings, video)
        overall_acc, per_class_acc = clustering_accuracy(gt_cluster_ids, predictions)
        return overall_acc, per_class_acc

    def stored_videos_clustering_accuracy(self, dataset_path: str, embeddings_directory: str, 
                                          clustering_method: MagicKmeansClustering) -> dict[str: Any]:
        '''Loads data to evaluate clustering'''
        if self.videos_list is None:
            self.videos_list = [video for video in os.listdir(dataset_path) if video[:5] != "SNGS-"]
        gt_reader = GroundTruthReader(dataset_path, self.videos_list)
        accuracy_dict = defaultdict(lambda: defaultdict(list))
        for video in tqdm(self.videos_list, total=len(self.videos_list)):
            # Get embeddings and annotations for this video in correct order
            groundtruth = []
            video_gt = gt_reader.get_annotations_from_video(video)
            video_embeddings_annotations_ids, video_embeddings = load_embeddings(embeddings_directory, video)
            for ann_id in video_embeddings_annotations_ids:
                for annotation in video_gt:
                    if annotation.annotation_id == ann_id:
                        groundtruth.append(annotation.role_id)
                        break
            # Perform clustering fit and eval with video embeddings and annotations
            overall_accuracy, clusters_accuracy_dict = self.video_clustering_accuracy(groundtruth, video_embeddings, 
                                                                                      clustering_method,
                                                                                      video)
            accuracy_dict[video]["overall"] = overall_accuracy
            accuracy_dict[video]["per_class"] = clusters_accuracy_dict
        average_acc_dict = get_accuracy_average(accuracy_dict)
        return average_acc_dict, accuracy_dict


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Script for clustering evaluation.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True, # example: ../../data/SoccerNetGS/train
        help="Dataset path: train, valid or test folder path.",
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
    args = vars(parser.parse_args())
    dataset_path = args["dataset_path"]
    videos_list = args["videos_list"]
    embeddings_directory = args["embeddings_directory"]

    clustering_evaluation = ClusteringAccuracyEvaluation(videos_list)
    clustering_method = MagicKmeansClustering()
    average_acc_dict, accuracy_dict = clustering_evaluation.stored_videos_clustering_accuracy(dataset_path,
                                                                                              embeddings_directory,
                                                                                              clustering_method)
    print(json.dumps(accuracy_dict, indent=4))
    print(json.dumps(average_acc_dict, indent=4))
