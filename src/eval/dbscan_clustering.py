import os
import json
import torch
import logging
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_embeddings(embeddings_directory: str, video_name: str) -> list[torch.FloatTensor]:
    '''Load embeddings from a given video'''
    embeddings_file_path = os.path.join(embeddings_directory, f"{video_name}_embeddings.npz")
    with np.load(embeddings_file_path) as data:
        model_path = None
        annotation_ids = data['annotation_ids']
        embeddings = data['embeddings']
        try:
            model_path = data['model_path'][0]
        except Exception as e:
            logger.debug(f"{str(e)} on {embeddings_file_path}")
    return model_path, annotation_ids, embeddings


class DBScanClustering():
    def __init__(self, eps: float=0.5, min_samples: int=5):
        self.db = DBSCAN(eps=eps, min_samples=min_samples)

    def fit(self, embeddings) -> list[int]:
        '''Clusters from given embeddings'''
        db = self.db.fit(embeddings)
        return db.labels_

    def export_predictions(self, video, annotations: list[int], predictions: list[int], output_directory: str) -> None:
        '''Given a video and its list of predictions (cluster indices), export them as a json file in a given directory.
            Each json file corresponds to a single video and its keys are annotation_ids while its values are the
            corresponding predicted role_id'''        
        output_path = os.path.join(output_directory, f"{video}_dbscan_predictions.json")
        output = defaultdict(list)
        for i in range(len(predictions)):
            output[annotations[i]] = int(predictions[i])
        os.makedirs(output_directory, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f)

    def cluster_and_export_games_list(self, games_list: list[str], embeddings_directory: str, output_directory: str, dataset_path: str) -> None:
        '''Load embeddings from a given directory and games_list, and export the predictions (clusters indices) 
            as a json file in a given directory'''
        for video in os.listdir(dataset_path):
            if games_list and video not in games_list:
                continue
            if video == ".ipynb_checkpoints":
                continue
            video_path = os.path.join(dataset_path, video)
            if not os.path.isdir(video_path):
                continue
            _, annotation_ids, embeddings = load_embeddings(embeddings_directory, video)
            predictions = self.fit(embeddings)
            self.export_predictions(video, annotation_ids, predictions, output_directory)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clustering script.")
    parser.add_argument(
        "--games_list",
        type=str,
        required=False,
        nargs="+",
        default=None,  # example from train: SNGS-060 SNGS-097 SNGS-103
        help="List of games to use for clustering.",
    )
    parser.add_argument(
        "--embeddings_directory",
        type=str,
        required=False,
        default="./video_files",
        help="Embeddings directory to get the .npz files.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=False,
        default="./video_files",
        help="Directory to save the json output file.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Dataset path of train, valid or test.",
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
    games_list = args["games_list"]
    embeddings_directory = args["embeddings_directory"]
    output_directory = args["output_directory"]
    dataset_path = args["dataset_path"]
    eps = args["eps"]
    min_samples = args["min_samples"]

    dbscan_clustering = DBScanClustering(eps=eps, min_samples=min_samples)
    dbscan_clustering.cluster_and_export_games_list(games_list, embeddings_directory, output_directory, dataset_path)
