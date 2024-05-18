import os
import sys
from pathlib import Path
import torch
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json
from itertools import chain
from collections import defaultdict

from sklearn.cluster import KMeans

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from SNGS_dataset.groundtruth_reader import GroundTruthReader, Annotation
from cluster_classification_dataset import Track
from eval.magic_kmeans_clustering import calculate_n_clusters
from real_clusters_dataset import BboxCluster, BBoxClustering, TrackCluster, TrackClustering, read_track_clustering_file, save_track_clustering_file

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


def bboxcluster_to_trackcluster(clusters: BBoxClustering, annotations: List[Annotation]) -> TrackClustering:
    # Maps each annotation ID to its corresponding track ID
    annotation_to_track = {ann.annotation_id: ann.track_id for ann in annotations}
    
    # Maps each track ID to its role ID
    track_to_role = {ann.track_id: ann.role_id for ann in annotations}

    # Initialize a mapping from track IDs to clusters based on majority vote
    track_to_cluster = defaultdict(lambda: defaultdict(int))
    for cluster in clusters.clusters:
        for annotation_id in cluster.annotation_ids:
            track_id = annotation_to_track.get(annotation_id)
            if track_id is not None:
                track_to_cluster[track_id][cluster.cluster_id] += 1

    # Determine the best cluster for each track by majority vote
    track_to_best_cluster = {}
    for track_id, cluster_votes in track_to_cluster.items():
        best_cluster = max(cluster_votes.items(), key=lambda x: x[1])[0]
        track_to_best_cluster[track_id] = best_cluster

    # Reverse mapping: from cluster IDs to lists of track IDs
    cluster_to_tracks = defaultdict(list)
    
    # Mapping to count votes for the most common role_id in each cluster
    cluster_to_role_votes = defaultdict(lambda: defaultdict(int))
    for track_id, cluster_id in track_to_best_cluster.items():
        cluster_to_tracks[cluster_id].append(track_id)
        cluster_to_role_votes[cluster_id][track_to_role[track_id]] += 1

    # Create the list of TrackCluster with the majority role_id when valid
    track_clusters = []
    for cluster_id, track_ids in cluster_to_tracks.items():
        if clusters.is_valid:
            # Get the role_id with the majority vote within this cluster
            most_common_role_id = max(cluster_to_role_votes[cluster_id].items(), key=lambda x: x[1])[0]
        else:
            # If the clustering is marked as invalid, set role_id to None
            most_common_role_id = None
        track_clusters.append(TrackCluster(cluster_id=cluster_id, track_ids=track_ids, gt_role_id=most_common_role_id))

    # Return the TrackClustering data class instance
    return TrackClustering(clusters=track_clusters, video=clusters.video, is_valid=clusters.is_valid)
    
    
def generate_bbox_clusters(video:str, dataset_path: str, embeddings_directory: str) -> List[BBoxClustering]:
    annotation_ids, embeddings = load_embeddings(embeddings_directory, video)
    for n_clusters in [3,4,5,6]:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit(embeddings).labels_
        # Convert cluster ids to letters to avoid problems
        cluster_ids = [chr(65 + cluster) for cluster in cluster_labels]
        
        clusters = np.unique(cluster_ids)

        bbox_clustering = BBoxClustering(
            clusters=[BboxCluster(
                    cluster_id=c,
                    annotation_ids=[annotation for annotation, cluster_id in zip(annotation_ids,cluster_ids) if cluster_id == c]
                ) for c in clusters],
            video=video,
            is_valid=n_clusters == calculate_n_clusters(video)
        )
        # Check all annotation should be inside a cluster
        all_annotation_ids_in_clusters = np.array(list(chain.from_iterable(c.annotation_ids for c in bbox_clustering.clusters)))
        assert np.array_equal(np.sort(all_annotation_ids_in_clusters), np.sort(annotation_ids))

        yield bbox_clustering
            
def make_real_clustering_dataset(dataset_path: str, embeddings_directory: str, out_path: str):
    # videos_list = ["SNGS-048"]
    videos_list = None
    gt_reader = GroundTruthReader(dataset_path, videos_list)   
    if videos_list is not None:
        videos_list =  [video for video in os.listdir(dataset_path) if video[:5] == "SNGS-" and video in videos_list]
    else:
        videos_list =  [video for video in os.listdir(dataset_path) if video[:5] == "SNGS-"]
    
    track_clusters = []
    for video in tqdm(videos_list):            
        video_annotations = gt_reader.get_annotations_from_video(video)
        for bbox_clustering in generate_bbox_clusters(video, dataset_path, embeddings_directory):
            track_cluster = bboxcluster_to_trackcluster(bbox_clustering, video_annotations)
            # Convert to json to save it
            track_clusters.append(track_cluster)
            
    save_track_clustering_file(track_clusters, out_path)

def get_stats(clusters_path: str, dataset_path: str):
    # videos_list = ["SNGS-048"]
    videos_list = None
    gt_reader = GroundTruthReader(dataset_path, videos_list)
    
    track_clusters = read_track_clustering_file(clusters_path)
        
    cluster_purities = []
    for tc in track_clusters:
        if tc.is_valid:
            video_annotations = gt_reader.get_annotations_from_video(tc.video)
            for c in tc.clusters:
                tracks_role_ids = [next((a.role_id for a in video_annotations if a.track_id == tid)) for tid in c.track_ids]
                # Cluster purity is how many tracks on the cluster are really that role_id given to the cluster
                cluster_purity = len([r for r in tracks_role_ids if r == c.gt_role_id]) / len(c.track_ids)
                cluster_purities.append(cluster_purity)
    # TODO: Make a plot or something like that with this
    print("Average purity", np.mean(cluster_purities))

        
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Classification model.")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="../../data/SoccerNetGS/valid",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        required=True,
        help="Path to a folder with the saved embeddings for videos",
    )
    parser.add_argument(
        "--out_dataset_path",
        type=str,
        required=True,
        help="Path to a json file where the dataset is going to be stored",
    )
    args = vars(parser.parse_args())

    # choose model type
    embeddings_path = args["embeddings_path"]
    data_path = args["data_path"]
    out_dataset_path = args["out_dataset_path"]
    
    make_real_clustering_dataset(data_path, embeddings_path, out_dataset_path)
    # get_stats(out_dataset_path, data_path)