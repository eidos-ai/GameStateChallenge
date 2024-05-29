import pandas as pd
import numpy as np
import torch
import sys
import os
import random
import string
import joblib
from typing import List
from collections import defaultdict, Counter
from dataclasses import dataclass
from tracklab.pipeline.videolevel_module import VideoLevelModule
from sklearn.cluster import KMeans

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from gsr.inspector import inn
from gsr.cluster_feats import ClusterFeats, Track, BBox

def get_role_and_team(pred_role_id: int):
    if pred_role_id == 0: # player_left
        return "player", 0, "left"
    elif pred_role_id == 1: # player_right
        return "player", 1, "right"
    elif pred_role_id == 2: # goalkeeper_left
        return "goalkeeper", None, "left"
    elif pred_role_id == 3: # goalkeeper_right
        return "goalkeeper", None, "right"
    elif pred_role_id == 4: # referee
        return "referee", None, None
    elif pred_role_id == 5: # other
        return "other", None, None
    else:
        return "player", 1, "right"

@dataclass
class ClassifiedTrack:
    track_id: int
    pred_role_id: int

@dataclass
class TrackCluster:
    cluster_id: str
    track_ids: List[int]

@dataclass
class TrackClustering:
    clusters: List[TrackCluster]
    video_tracks: List[Track] # 

# Runs the whole cluster classification pipeline
class ClusterPredictionModel():
    def __init__(self, prediction_model):
        self.prediction_model = prediction_model
        self.cluster_feats = ClusterFeats()

    # Predicts most likely role_id for the cluster, but ruling out a invalid clustering
    def predict_cluster_label(self, cluster_tracks : List[Track], rest_tracks: List[Track]) -> float:
        feats = self.cluster_feats.forward(cluster_tracks, rest_tracks)
        feats_vector = np.array([feats.to_vector()])
        pred_proba = self.prediction_model.predict_proba(feats_vector)[0]
        # This method wont return invalid track
        probs_role_ids = pred_proba[:-1]
        highest_prob_class = np.argmax(probs_role_ids)
        return highest_prob_class

    # Predicts the probability of this cluster being a valid cluster
    def predict_score_cluster_valid(self, cluster_tracks : List[Track], rest_tracks: List[Track]) -> float:
        feats = self.cluster_feats.forward(cluster_tracks, rest_tracks)
        feats_vector = np.array([feats.to_vector()])
        pred_proba = self.prediction_model.predict_proba(feats_vector)
        # The probability of being valid is 1 - prob being invalid
        return 1 - pred_proba[0][6]

def predict_cluster_valid_score(cluster: TrackCluster, model, video_tracks: List[Track]):
    cluster_tracks = [t for t in video_tracks if t.track_id in cluster.track_ids]
    rest_tracks = [t for t in video_tracks if t.track_id not in cluster.track_ids]

    score_cluster = model.predict_score_cluster_valid(cluster_tracks, rest_tracks)

    return score_cluster

def predict_cluster_class(cluster: TrackCluster, model, video_tracks: List[Track]):
    cluster_tracks = [t for t in video_tracks if t.track_id in cluster.track_ids]
    rest_tracks = [t for t in video_tracks if t.track_id not in cluster.track_ids]

    pred_cluster_class = model.predict_cluster_label(cluster_tracks, rest_tracks)

    return pred_cluster_class


def eval_clustering_confidence(clustering: TrackClustering, model) -> float:
    video_tracks: List[Track] = clustering.video_tracks
    clustering_valid_scores = [predict_cluster_valid_score(cluster, model, video_tracks) for cluster in clustering.clusters]

    # All have to be valid
    # clustering_score = reduce(operator.mul, clustering_valid_scores , 1)

    # Mean of being valid
    # clustering_score = np.mean(clustering_valid_scores)

    # Weighted mean of being valid
    clustering_valid_scores = [len(cluster.track_ids) * predict_cluster_valid_score(cluster, model, video_tracks) for cluster in clustering.clusters]
    total_cluster_size = sum(len(cluster.track_ids) for cluster in clustering.clusters)
    clustering_score = np.sum(clustering_valid_scores) / total_cluster_size

    return clustering_score

def cluster_to_tracks(clustering: TrackClustering, model) -> List[ClassifiedTrack]:
    video_tracks: List[Track] = clustering.video_tracks
    classified_tracks: List[ClassifiedTrack] = []
    for cluster in clustering.clusters:
        cluster_class = predict_cluster_class(cluster, model, video_tracks)
        for track_id in cluster.track_ids:
            classified_tracks.append(ClassifiedTrack(
                track_id=track_id,
                pred_role_id=cluster_class
            ))
    return classified_tracks

def generate_unique_capital_letters(length):
    if length > 26:
        raise ValueError("Length cannot be more than 26 as there are only 26 unique capital letters.")
    letters = list(string.ascii_uppercase)
    random.shuffle(letters)
    return letters[:length]

def get_video_tracks(detections: pd.DataFrame) -> List[Track]:
    video_tracks: List[Track] = []
    bboxes_per_track_id = defaultdict(list) # {track_id: List[Bbox]}
    for i in range(len(detections)):
        bboxes_per_track_id[detections["track_id"].iloc[i]]
        bbox_frame = str(detections["image_id"].iloc[i])[-6:]
        bbox_x_image = detections["bbox_ltwh"].iloc[i][0]
        bbox_y_image = detections["bbox_ltwh"].iloc[i][1]
        bbox_w_image = detections["bbox_ltwh"].iloc[i][2]
        bbox_h_image = detections["bbox_ltwh"].iloc[i][3]
        bbox_x_pitch = None
        bbox_y_pitch = None
        try:
            bbox_x_pitch = detections["bbox_pitch"].iloc[i]["x_bottom_middle"]
            bbox_y_pitch = detections["bbox_pitch"].iloc[i]["y_bottom_middle"]
        except TypeError:
            bbox_x_pitch = detections["bbox_pitch"].iloc[i]
            bbox_y_pitch = detections["bbox_pitch"].iloc[i]
        bbox = BBox(frame=bbox_frame ,x_image=bbox_x_image, y_image=bbox_y_image, w_image=bbox_w_image, h_image=bbox_h_image, x_pitch=bbox_x_pitch, y_pitch=bbox_y_pitch)
        bboxes_per_track_id[detections["track_id"].iloc[i]].append(bbox)
    for track_id, track_bboxes in bboxes_per_track_id.items():
        track = Track(track_id=track_id, bboxes=track_bboxes)
        video_tracks.append(track)
    return video_tracks


class RoleClassification(VideoLevelModule):
    input_columns = ["role_embeddings", "track_id"]
    output_columns = ["role", "team_cluster", "team"]

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        prediction_model = joblib.load(self.cfg.model_path)
        self.model = ClusterPredictionModel(prediction_model)

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        num_clusters = [3, 4, 5, 6]
        kmeans_clusterings = defaultdict(list)
        for n in num_clusters:
            embeddings_2d = np.vstack(detections["role_embeddings"].values)
            kmeans = KMeans(n_clusters=n, random_state=0).fit(embeddings_2d)
            kmeans_clusterings[n] = kmeans.labels_

        track_clusterings: List[TrackClustering] = []

        # get all Track instances of the video and save them in video_tracks: List[Track]
        video_tracks = get_video_tracks(detections)

        # use kmeans for n_clusters = [3, 4, 5, 6]
        for kmeans in kmeans_clusterings.keys():
            predicted_clusters = kmeans_clusterings[kmeans]
            track_ids = detections["track_id"].tolist()
            assert len(track_ids) == len(predicted_clusters) # chequeo que haya tantos track_ids como predicciones

            tracks_predictions = defaultdict(list) # creo un dict {track_id: most_predicted_cluster_id}
            for track_idx, track_id in enumerate(track_ids):
                if np.isnan(track_id):
                    continue
                temp_track_predictions: List[int] = []
                for pred_idx, prediction in enumerate(predicted_clusters):
                    if track_ids[pred_idx] == track_id:
                        temp_track_predictions.append(prediction)
                tracks_predictions[track_id] = Counter(temp_track_predictions).most_common(1)[0][0]

            # generate a list of TrackCluster objects
            clusters = defaultdict(list)
            for track_id, predicted_cluster_id in tracks_predictions.items():
                clusters[predicted_cluster_id].append(track_id)
            track_clusters: List[TrackCluster] = []
            alpha_cluster_id = generate_unique_capital_letters(len(clusters))
            for idx, track_ids in enumerate(clusters.values()):
                track_clusters.append(TrackCluster(cluster_id=alpha_cluster_id[idx], track_ids=track_ids))

            # create a TrackClustering object
            track_clustering = TrackClustering(clusters=track_clusters, video_tracks=video_tracks)
            track_clusterings.append(track_clustering)

        clustering_confidences = [eval_clustering_confidence(c, self.model) for c in track_clusterings]
        best_clustering: TrackClustering = track_clusterings[clustering_confidences.index(max(clustering_confidences))]
        track_predictions = cluster_to_tracks(best_clustering, self.model) # List[ClassifiedTrack(track_id, pred_role_id]

        print("///////")
        print(best_clustering.clusters)
        print("///////")
        print([classified_track.pred_role_id for classified_track in track_predictions])
        print("///////")

        detections['team_cluster'] = None
        detections['team'] = None
        detections['role'] = None
        track_id_to_pred_role_id = {ct.track_id: ct.pred_role_id for ct in track_predictions}
        for idx, row in detections.iterrows():
            if row['track_id'] in track_id_to_pred_role_id:
                role, team_cluster, side = get_role_and_team(track_id_to_pred_role_id[row['track_id']])
                detections.at[idx, 'role'] = role
                detections.at[idx, 'team_cluster'] = team_cluster
                detections.at[idx, 'team'] = side

        player_detections = detections[detections['role'] == 'player']
        role_embeddings = player_detections['role_embeddings'].tolist()
        kmeans = KMeans(n_clusters=2, random_state=0).fit(role_embeddings)
        player_detections['team_cluster'] = kmeans.labels_

        # mapeo track_id con el equipo al que es asignado más veces
        track_ids = detections["track_id"].tolist()
        tracks_predictions = defaultdict(list) # creo un dict {track_id: most_predicted_cluster_id}
        for track_idx, track_id in enumerate(track_ids):
            if np.isnan(track_id):
                continue
            temp_track_predictions: List[int] = []
            for pred_idx, prediction in enumerate(player_detections['team_cluster']):
                if track_ids[pred_idx] == track_id:
                    temp_track_predictions.append(prediction)
            tracks_predictions[track_id] = Counter(temp_track_predictions).most_common(1)[0][0]

        print(tracks_predictions)

        # itero en el df y cuando aparece un track_id, le asigno el team y team_cluster de ese track_id
        for det_idx, det_row in detections.iterrows():
            if det_row['track_id'] in tracks_predictions:
                detections.at[det_idx, 'team_cluster'] = tracks_predictions[det_row['track_id']]
                if tracks_predictions[det_row['track_id']] == 0:
                    detections.at[idx, 'team'] = 'left'
                else:
                    detections.at[idx, 'team'] = 'right'

        return detections
