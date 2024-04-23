from collections import defaultdict
import sys
sys.path.append("..")
sys.path.append("../SNGS_dataset")
from SNGS_dataset.groundtruth_reader import GroundTruthReader
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np


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


class ClusteringEvaluation():
    def __init__(self, dataset_path: str, games_list: list[str], predictions_directory: str):
        self.dataset_path = dataset_path
        self.games_list = games_list
        self.predictions_directory = predictions_directory

    def eval_clustering(self) -> float:
        '''Evaluate clusters predictions against ground truth labels.'''
        gt_reader = GroundTruthReader(self.dataset_path, self.games_list)
        accuracy_dict = defaultdict()
        for video in self.games_list:
            video_annotations = gt_reader.get_annotations_from_video(video)
            groundtruth = []
            for annotation in video_annotations:
                groundtruth.append(annotation["role_id"]) # list[role_id]
            predictions = np.load(f"{self.predictions_directory}/SNGS-{video}_predictions.json").values() # list[cluster_id]
            overall_acc, per_class_acc = clustering_accuracy(groundtruth, predictions)
        # promediar acc por video, y por clase (sobre todos los videos)


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
        default=[],
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
