import os
import json
import numpy as np
import torch
import logging
from collections import defaultdict
from sklearn.cluster import KMeans

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_embeddings(
    embeddings_directory: str, video_name: str
) -> list[torch.FloatTensor]:
    """Load embeddings from a given video"""
    embeddings_file_path = os.path.join(
        embeddings_directory, f"{video_name}_embeddings.npz"
    )
    with np.load(embeddings_file_path) as data:
        try:
            model_path = data["model_path"][0]
        except Exception as e:
            logger.debug(f"{str(e)} on {embeddings_file_path}")
        annotation_ids = data["annotation_ids"]
        embeddings = data["embeddings"]
    return annotation_ids, embeddings


def calculate_n_clusters(video):
    role_id_per_video = {  # train videos
        "SNGS-157": 4,
        "SNGS-155": 4,
        "SNGS-101": 5,
        "SNGS-152": 4,
        "SNGS-100": 5,
        "SNGS-071": 4,
        "SNGS-164": 4,
        "SNGS-113": 5,
        "SNGS-114": 4,
        "SNGS-074": 4,
        "SNGS-070": 5,
        "SNGS-162": 5,
        "SNGS-076": 5,
        "SNGS-067": 5,
        "SNGS-160": 4,
        "SNGS-108": 5,
        "SNGS-115": 4,
        "SNGS-112": 4,
        "SNGS-063": 5,
        "SNGS-102": 4,
        "SNGS-069": 3,
        "SNGS-065": 5,
        "SNGS-099": 5,
        "SNGS-098": 5,
        "SNGS-165": 5,
        "SNGS-111": 5,
        "SNGS-158": 5,
        "SNGS-110": 4,
        "SNGS-170": 5,
        "SNGS-061": 5,
        "SNGS-072": 4,
        "SNGS-167": 4,
        "SNGS-103": 4,
        "SNGS-161": 5,
        "SNGS-168": 4,
        "SNGS-064": 4,
        "SNGS-068": 4,
        "SNGS-073": 5,
        "SNGS-156": 4,
        "SNGS-166": 5,
        "SNGS-066": 4,
        "SNGS-154": 4,
        "SNGS-107": 4,
        "SNGS-151": 5,
        "SNGS-163": 5,
        "SNGS-060": 5,
        "SNGS-109": 4,
        "SNGS-105": 4,
        "SNGS-062": 4,
        "SNGS-159": 4,
        "SNGS-153": 5,
        "SNGS-077": 4,
        "SNGS-097": 4,
        "SNGS-075": 4,
        "SNGS-106": 4,
        "SNGS-169": 5,
        "SNGS-104": 4,
        # valid videos
        "SNGS-092": 5,
        "SNGS-029": 4,
        "SNGS-051": 4,
        "SNGS-040": 4,
        "SNGS-087": 5,
        "SNGS-034": 4,
        "SNGS-088": 4,
        "SNGS-091": 5,
        "SNGS-059": 4,
        "SNGS-082": 4,
        "SNGS-095": 4,
        "SNGS-056": 4,
        "SNGS-096": 5,
        "SNGS-023": 4,
        "SNGS-026": 5,
        "SNGS-022": 4,
        "SNGS-048": 4,
        "SNGS-030": 4,
        "SNGS-044": 5,
        "SNGS-038": 4,
        "SNGS-045": 5,
        "SNGS-053": 4,
        "SNGS-086": 4,
        "SNGS-094": 5,
        "SNGS-031": 4,
        "SNGS-055": 3,
        "SNGS-078": 5,
        "SNGS-042": 4,
        "SNGS-080": 4,
        "SNGS-039": 4,
        "SNGS-043": 5,
        "SNGS-058": 5,
        "SNGS-041": 4,
        "SNGS-057": 4,
        "SNGS-090": 4,
        "SNGS-083": 5,
        "SNGS-084": 5,
        "SNGS-046": 4,
        "SNGS-054": 5,
        "SNGS-081": 5,
        "SNGS-027": 4,
        "SNGS-079": 5,
        "SNGS-037": 4,
        "SNGS-085": 4,
        "SNGS-028": 4,
        "SNGS-050": 4,
        "SNGS-093": 5,
        "SNGS-052": 4,
        "SNGS-033": 4,
        "SNGS-036": 6,
        "SNGS-049": 4,
        "SNGS-089": 5,
        "SNGS-047": 4,
        "SNGS-032": 5,
        "SNGS-035": 4,
        "SNGS-025": 4,
        "SNGS-021": 4,
        "SNGS-024": 4,
    }
    return role_id_per_video[video]


class MagicKmeansClustering:
    def __init__(self):
        self.clustering = KMeans(n_clusters=5, random_state=0)

    def fit(
        self, embeddings: dict[str, int | torch.FloatTensor], video: str
    ) -> list[int]:
        """Clusters from given embeddings"""
        number_of_clusters = calculate_n_clusters(video)
        kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
        clusters = kmeans.fit(embeddings)
        return clusters.labels_

    def export_predictions(
        self,
        video,
        annotations: list[int],
        predictions: list[int],
        output_directory: str,
    ) -> None:
        """Given a video and its list of predictions (cluster indices), export them as a json file in a given directory.
        Each json file corresponds to a single video and its keys are annotation_ids while its values are the
        corresponding predicted role_id"""
        output_path = os.path.join(output_directory, f"{video}_predictions.json")
        output = defaultdict(list)
        for i in range(len(predictions)):
            output[annotations[i]] = int(predictions[i])
        os.makedirs(output_directory, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f)

    def cluster_and_export_games_list(
        self,
        videos_list: list[str],
        embeddings_directory: str,
        output_directory: str,
        dataset_path: str,
    ) -> None:
        """Load embeddings from a given directory and games_list, and export the predictions (clusters indices)
        as a json file in a given directory"""
        for video in os.listdir(dataset_path):
            if videos_list and video not in videos_list:
                continue
            if video == ".ipynb_checkpoints":
                continue
            video_path = os.path.join(dataset_path, video)
            if not os.path.isdir(video_path):
                continue
            annotation_ids, embeddings = load_embeddings(embeddings_directory, video)
            predictions = self.fit(embeddings, video)
            self.export_predictions(
                video, annotation_ids, predictions, output_directory
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clustering script.")
    parser.add_argument(
        "--videos_list",
        type=str,
        required=False,
        nargs="+",
        default=None,  # example from train: SNGS-060 SNGS-097 SNGS-103
        help="List of videos to use for clustering.",
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
        "--data_path",
        type=str,
        required=True,
        help="Path of train, valid or test.",
    )

    args = vars(parser.parse_args())
    videos_list = args["videos_list"]
    embeddings_directory = args["embeddings_directory"]
    output_directory = args["output_directory"]
    data_path = args["data_path"]

    magic_kmeans = MagicKmeansClustering()
    magic_kmeans.cluster_and_export_games_list(
        videos_list, embeddings_directory, output_directory, data_path
    )
