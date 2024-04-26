from dataclasses import asdict
import os
import sys
import numpy as np
sys.path.append("..")
sys.path.append("../SNGS_dataset")
from SNGS_dataset.groundtruth_reader import GroundTruthReader
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from typing import Any
from clustering import load_embeddings
from clustering import TeamClustering


def clustering_accuracy(labels_true, labels_pred):
    # Create the confusion matrix
    cm = confusion_matrix(labels_true, labels_pred, labels=[0, 1, 2, 3, 4])
    print(cm)

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
        per_class_accuracy[row] = class_accuracy

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


class ClusteringEvaluation():
    def __init__(self, games_list: list[str]):
        self.games_list = games_list

    def videos_clustering_accuracy(self, annotations: list[dict[str: Any]], embeddings: list[Any], teamClustering: TeamClustering) -> dict[str: Any]:
        '''Evaluates clustering from on-memory data'''
        assert len(annotations) == len(embeddings), "The annotations and embeddings lists have different lengths"
        accuracy_dict = defaultdict(lambda: defaultdict(list))
        for video in self.games_list:
            video_embeddings = []
            video_annotations = []
            for idx, annotation in enumerate(annotations):
                if annotation["video"] == video:
                    video_embeddings.append(embeddings[idx])
                    video_annotations.append(annotation["role_id"])
            predictions = teamClustering.fit(video_embeddings, video)
            overall_acc, per_class_acc = clustering_accuracy(video_annotations, predictions)
            accuracy_dict[video]["overall"] = overall_acc
            accuracy_dict[video]["per_class"] = defaultdict()
            classes_acc = accuracy_dict[video]["per_class"]
            for class_idx in per_class_acc.keys():
                classes_acc[int(class_idx)] = per_class_acc[class_idx]
        average_acc_dict = get_accuracy_average(accuracy_dict)
        return average_acc_dict, accuracy_dict

    def stored_videos_clustering_accuracy(self, dataset_path: str, embeddings_directory: str) -> dict[str: Any]:
        '''Loads data to evaluate clustering'''
        if self.games_list is None:
            self.games_list = [video for video in os.listdir(dataset_path)]
        gt_reader = GroundTruthReader(dataset_path, self.games_list)
        
        groundtruth = []
        embeddings = []
        for video in self.games_list:
            if video[:5] != "SNGS-":
                continue
            print(f"Now evaluating video {video}")

            video_gt = gt_reader.get_annotations_from_video(video)
            video_embeddings_annotations_ids, video_embeddings = load_embeddings(embeddings_directory, video)
            for ann_id, embedding in zip(video_embeddings_annotations_ids, video_embeddings):
                embeddings.append(embedding)
                for annotation in video_gt:
                    if annotation.annotation_id == ann_id:
                        groundtruth.append(asdict(annotation))
                        break
        team_clustering = TeamClustering()
        average_acc_dict, accuracy_dict = self.videos_clustering_accuracy(groundtruth, embeddings, team_clustering)
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
        "--games_list",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="List of games to use for clustering evaluation",
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
    games_list = args["games_list"]
    embeddings_directory = args["embeddings_directory"]

    clustering_evaluation = ClusteringEvaluation(games_list)
    average_acc_dict, accuracy_dict = clustering_evaluation.stored_videos_clustering_accuracy(dataset_path,
                                                                                              embeddings_directory)
    print(json.dumps(accuracy_dict, indent=4))
    print(json.dumps(average_acc_dict, indent=4))
