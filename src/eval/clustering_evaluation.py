import os
import sys
import json
import numpy as np
sys.path.append("..")
sys.path.append("../SNGS_dataset")
from SNGS_dataset.groundtruth_reader import GroundTruthReader
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from collections import defaultdict


def clustering_accuracy(labels_true, labels_pred):
    # Create the confusion matrix
    cm = confusion_matrix(labels_true, labels_pred)

    # Find the optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)

    # Calculate the overall accuracy using the optimal assignment
    overall_accuracy = cm[row_ind, col_ind].sum() / cm.sum()

    # Calculate per-class accuracy
    per_class_accuracy = {}
    for i, (row, col) in enumerate(zip(row_ind, col_ind)):
        class_accuracy = cm[row, col] / np.sum(cm[row, :])
        per_class_accuracy[row] = class_accuracy

    return overall_accuracy, per_class_accuracy

def get_accuracy_average(accuracy_dict):
    '''Get the average accuracy of all videos.'''
    overall_acc = 0
    per_class_acc = defaultdict(float)
    for video in accuracy_dict.keys():
        overall_acc += accuracy_dict[video]["overall"]
        for class_idx in accuracy_dict[video]["per_class"].keys():
            per_class_acc[int(class_idx)] += accuracy_dict[video]["per_class"][class_idx]
    overall_acc /= len(accuracy_dict)
    for class_idx in per_class_acc.keys():
        per_class_acc[class_idx] /= len(accuracy_dict)
    avg_dict = defaultdict(lambda: defaultdict(list))
    avg_dict["overall"] = overall_acc
    avg_dict["per_class"] = per_class_acc
    return avg_dict


class ClusteringEvaluation():
    def __init__(self, dataset_path: str, games_list: list[str], predictions_directory: str):
        self.dataset_path = dataset_path
        self.games_list = games_list
        self.predictions_directory = predictions_directory

    def clustering_accuracy(self) -> float:
        '''Evaluate clusters predictions against ground truth labels.'''
        gt_reader = GroundTruthReader(self.dataset_path, self.games_list)
        accuracy_dict = defaultdict(lambda: defaultdict(list))
        if self.games_list is None:
            self.games_list = [video for video in os.listdir(dataset_path)]
        for video in self.games_list:
            if video[:5] != "SNGS-":
                continue
            video_annotations = gt_reader.get_annotations_from_video(video)
            groundtruth = []
            for annotation in video_annotations:
                groundtruth.append(annotation["role_id"]) # list[role_id]
            with open(os.path.join(self.predictions_directory, f"{video}_predictions.json"), "r") as f:
                predictions = list(json.load(f).values()) # list[cluster_id]

            # get accuracies
            overall_acc, per_class_acc = clustering_accuracy(groundtruth, predictions)

            # load them into a dictionary
            accuracy_dict[video]["overall"] = overall_acc
            accuracy_dict[video]["per_class"] = defaultdict()
            classes_acc = accuracy_dict[video]["per_class"]
            for class_idx in per_class_acc.keys():
                classes_acc[int(class_idx)] = per_class_acc[class_idx]
        average_acc_dict = get_accuracy_average(accuracy_dict)

        print(json.dumps(accuracy_dict, indent=4))
        print(json.dumps(average_acc_dict, indent=4))
        return average_acc_dict, accuracy_dict


if __name__ == "__main__":
    import argparse

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
        "--predictions_directory",
        type=str,
        required=False,
        default="./video_files",
        help="Directory where the 'SNGS-XXX_predictions.json' files are.",
    )
    args = vars(parser.parse_args())
    dataset_path = args["dataset_path"]
    games_list = args["games_list"]
    predictions_directory = args["predictions_directory"]

    clustering_evaluation = ClusteringEvaluation(dataset_path, games_list, predictions_directory)
    clustering_evaluation.clustering_accuracy()
