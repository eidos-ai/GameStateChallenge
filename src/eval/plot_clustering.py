import os
import sys
import numpy as np
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from magic_kmeans_clustering import MagicKmeansClustering, load_embeddings
from dbscan_clustering import DBScanClustering
from SNGS_dataset.groundtruth_reader import GroundTruthReader
from collections import defaultdict
from sklearn.cluster import KMeans
from typing import List, Dict
from tqdm import tqdm


def get_labels_coordinates(embeddings_directory: str, videos_list: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    labels_coordinates = defaultdict(lambda: defaultdict(list))
    for video in tqdm(videos_list, total=len(videos_list), desc="Videos iteration to get embeddings"):
        annotation_ids, embeddings = load_embeddings(embeddings_directory, video)
        for i in range(len(annotation_ids)):
            labels_coordinates[video][annotation_ids[i]] = embeddings[i]
    return labels_coordinates

def get_groundtruth_role_id(data_path: str, videos_list: List[str], labels_coordinates: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, int]]:
    gt_role_id = defaultdict(lambda: defaultdict(list))
    gt_reader = GroundTruthReader(data_path, videos_list)
    for video in tqdm(videos_list, total=len(videos_list), desc="Videos iteration to get groundtruth role_id"):
        video_annotations = gt_reader.get_annotations_from_video(video)
        for key in labels_coordinates[video].keys():
            for gt_annotation in video_annotations:
                if key == gt_annotation.annotation_id:
                    gt_role_id[video][key] = gt_annotation.role_id
                    break
    return gt_role_id

def get_magic_kmeans_role_id(embeddings_directory: str, videos_list: List[str]) -> Dict[str, Dict[str, int]]:
    magic_kmeans_role_id = defaultdict(lambda: defaultdict(list))
    magic_kmeans = MagicKmeansClustering()
    for video in tqdm(videos_list, total=len(videos_list), desc="Videos iteration to get Magic Kmeans role_id"):
        annotation_ids, embeddings = load_embeddings(embeddings_directory, video)
        predictions = magic_kmeans.fit(embeddings, video)
        for i in range(len(annotation_ids)):
            magic_kmeans_role_id[video][annotation_ids[i]] = predictions[i]
    return magic_kmeans_role_id

def get_kmeans_role_id(embeddings_directory: str, videos_list: List[str], number_of_clusters: int) -> Dict[str, Dict[str, int]]:
    kmeans_role_id = defaultdict(lambda: defaultdict(list))
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
    for video in tqdm(videos_list, total=len(videos_list), desc="Videos iteration to get Kmeans role_id"):
        annotation_ids, embeddings = load_embeddings(embeddings_directory, video)
        results = kmeans.fit(embeddings, video)
        predictions = results.labels_
        for i in range(len(annotation_ids)):
            kmeans_role_id[video][annotation_ids[i]] = predictions[i]
    return kmeans_role_id

def get_dbscan_role_id(embeddings_directory: str, videos_list: List[str], eps: float=0.5, min_samples: int=5) -> Dict[str, Dict[str, int]]:
    dbscan_role_id = defaultdict(lambda: defaultdict(list))
    dbscan = DBScanClustering(eps=eps, min_samples=min_samples)
    for video in tqdm(videos_list, total=len(videos_list), desc="Videos iteration to get DBScan role_id"):
        annotation_ids, embeddings = load_embeddings(embeddings_directory, video)
        predictions = dbscan.fit(embeddings)
        for i in range(len(annotation_ids)):
            dbscan_role_id[video][annotation_ids[i]] = predictions[i]
    return dbscan_role_id
