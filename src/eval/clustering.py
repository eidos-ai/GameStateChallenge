import os
import json
import numpy as np
import torch
from collections import defaultdict
from sklearn.cluster import KMeans


class TeamClustering():
    def __init__(self, clustering_method: str="kmeans", n_clusters: int=5):
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        if clustering_method == "kmeans":
            self.clustering = KMeans(n_clusters=self.n_clusters, random_state=0)

    def load_embeddings(self, embeddings_directory: str, video_name: str) -> list[torch.FloatTensor]:
        '''Load embeddings from a given video'''
        embeddings_file_path = os.path.join(embeddings_directory, f"{video_name}_embeddings.npz")
        with np.load(embeddings_file_path) as data:
            annotation_ids = data['annotation_ids']
            embeddings = data['embeddings']
        return annotation_ids, embeddings

    def fit(self, embeddings: dict[str, int | torch.FloatTensor]) -> list[int]:
        '''Clusters from given embeddings'''
        clusters = self.clustering.fit(embeddings)
        return clusters.labels_
    
    def export_predictions(self, video, annotations: list[int], predictions: list[int], output_directory: str) -> None:
        '''Given a video and its list of predictions (cluster indices), export them as a json file in a given directory.
            Each json file corresponds to a single video and its keys are annotation_ids while its values are the
            corresponding predicted role_id'''        
        output_path = os.path.join(output_directory, f"{video}_predictions.json")
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
            annotation_ids, embeddings = self.load_embeddings(embeddings_directory, video)
            predictions = self.fit(embeddings)
            self.export_predictions(video, annotation_ids, predictions, output_directory)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clustering script.")
    parser.add_argument(
        "--clustering_method",
        type=str,
        required=False,
        default="kmeans",
        help="Clustering method. e.g. kmeans, dbscan, etc.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        required=False,
        default=5,
        help="Number of clusters.",
    )
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

    args = vars(parser.parse_args())
    clustering_method = args["clustering_method"]
    n_clusters = args["n_clusters"]
    games_list = args["games_list"]
    embeddings_directory = args["embeddings_directory"]
    output_directory = args["output_directory"]
    dataset_path = args["dataset_path"]

    team_clustering = TeamClustering(clustering_method, n_clusters)
    team_clustering.cluster_and_export_games_list(games_list, embeddings_directory, output_directory, dataset_path)
