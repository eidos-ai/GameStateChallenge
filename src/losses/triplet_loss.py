from itertools import combinations
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def distance(tensor_a, tensor_b):
    return (tensor_a - tensor_b).pow(2).sum(1)

def pdist(vectors: torch.FloatTensor):
    """
    Returns the computed pairwise distance matrix of a 2D tensor, where each element represents the squared Euclidean distance between a pair of vectors from the input.
    ||a - b||^2 = ||a||^2 + ||b||^2 -2a.b
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def pdist2(A: torch.FloatTensor, B: torch.FloatTensor):
    """
    Returns the computed pairwise distance matrix of two 1D tensors, where each element represents the squared Euclidean distance between the pair of the 1D tensors.
    ||a - b||^2 = ||a||^2 + ||b||^2 -2a.b
    """
    product_term = -2 * A.mm(torch.t(B))
    sum_sq_A = A.pow(2).sum(dim=1).view(-1, 1)  # Reshape to a column vector
    sum_sq_B = B.pow(2).sum(dim=1).view(1, -1)  # Reshape to a row vector
    distance_matrix = product_term + sum_sq_A + sum_sq_B
    return distance_matrix

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings: torch.FloatTensor, labels: torch.IntTensor, videos: List[str]):
        raise NotImplementedError

class SoccerPersonTripletSelector(TripletSelector):
    """
    Selector for anchors as brands.
    For every image, the positive anchor is the Brand Definition Logo/Icon
    For every image, the negative anchors are the Differrent Brand Definition Logos/Icons
    """
    def __init__(self, margin, negative_selection_fn, cpu = True):
        super(SoccerPersonTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn
        
    
    def get_triplets(self, embeddings: torch.FloatTensor, labels: torch.LongTensor, videos: List[str]):
        embeddings = embeddings.cpu()
        labels = labels.cpu().numpy()
        # Create matrix with squared euclidean distances of all embeddings with each other
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
                
            negative_indices = np.where(np.logical_not(label_mask))[0]

            # Positive pairs are same role_id, same video
            # Group indices by video
            video_dict = {}
            for index in label_indices:
                video_name = videos[index]
                if video_name not in video_dict:
                    video_dict[video_name] = []
                video_dict[video_name].append(index)
            
            # Triplets are made within each video.
            for video_name, video_indices in video_dict.items():
                if len(video_indices) < 2:
                    continue
                combinations_indices = list(combinations(video_indices, 2))
                
                for anchor_idx, positive_idx in combinations_indices:
                    # For negatives, select candidates with different labels within the same video
                    other_label_indices = [idx for idx in negative_indices if videos[idx] == video_name]
                    
                    if not other_label_indices:
                        continue
                        
                    # Compute loss values for each negative candidate
                    ap_distance = distance_matrix[anchor_idx, positive_idx]
                    loss_values = ap_distance - distance_matrix[anchor_idx, np.array(other_label_indices)] + self.margin
                    loss_values = loss_values.data.cpu().numpy()
                    
                    # Select the hardest negative using the provided selection function
                    hard_negative = self.negative_selection_fn(loss_values)
                    if hard_negative is not None:
                        hard_negative_idx = other_label_indices[hard_negative]
                        triplets.append([anchor_idx, positive_idx, hard_negative_idx])
                        
        triplets = np.array(triplets)
        return torch.LongTensor(triplets)

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector: SoccerPersonTripletSelector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target, videos):
        
        triplets = self.triplet_selector.get_triplets(embeddings, target, videos)
        if len(triplets) > 0:
            valid_loss = True
            if embeddings.is_cuda:
                triplets = triplets.cuda()

            # Triplets are given as image_index, anchor_positive_index, anchor_negative_index.
            # So last indexes are taken from anchors_embeddings
            ap_distances = distance(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]])
            an_distances = distance(embeddings[triplets[:, 0]], embeddings[triplets[:, 2]])
            losses = F.relu(ap_distances - an_distances + self.margin)

            ap_distances_mean = ap_distances.mean().item()
            an_distances_mean = an_distances.mean().item()
            mean_loss = losses.mean()
            n_triplets = len(triplets)
        else:
            valid_loss = False
            mean_loss = None
            n_triplets = 0
            ap_distances_mean = None
            an_distances_mean = None
            
        return valid_loss, mean_loss, n_triplets, ap_distances_mean, an_distances_mean


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings: torch.FloatTensor, labels: torch.IntTensor):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values: np.ndarray):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None        


def random_hard_negative(loss_values: np.ndarray):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values: np.ndarray, margin: float):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


def HardestNegativeTripletSelector(margin, cpu=False): 
    return SoccerPersonTripletSelector(margin=margin,
                                       negative_selection_fn=hardest_negative,
                                       cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): 
    return SoccerPersonTripletSelector(margin=margin,
                                       negative_selection_fn=random_hard_negative,
                                       cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): 
    return SoccerPersonTripletSelector(margin=margin,
                                       negative_selection_fn=lambda x: semihard_negative(x, margin),
                                       cpu=cpu)
