from itertools import combinations
import logging

import numpy as np
import torch


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pdist(vectors: torch.FloatTensor):
    """
    Returns the computed pairwise distance matrix, where each element represents the squared Euclidean distance between a pair of vectors from the input.
    ||a - b||^2 = ||a||^2 + ||b||^2 -2a.b
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def pdist2(A: torch.FloatTensor, B: torch.FloatTensor):
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

    def get_triplets(self, embeddings: torch.FloatTensor, labels: torch.IntTensor):
        raise NotImplementedError


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

class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings: torch.LongTensor, labels: torch.IntTensor):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []
        # iterate over unique labels
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0] # indices of negative elements to current label
            anchor_positives = list(combinations(label_indices, 2)) # indices of all anchor-positive pairs to current label
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

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
        
    def get_triplets(self, embeddings, anchors_embeddings, labels):
        logger.debug(f"Triplet selector embeddings {embeddings.shape}")
        logger.debug(f"Triplet selector anchors_embeddings {anchors_embeddings.shape}")
        logger.debug(f"Triplet selector labels {labels.shape}")
        
        if self.cpu:
            embeddings = embeddings.cpu()
            anchors_embeddings = anchors_embeddings.cpu()
        
        distance_matrix = pdist2(embeddings, anchors_embeddings)
        distance_matrix = distance_matrix.cpu()
        logger.debug(f"distance_matrix {distance_matrix.shape}")
        
        labels = labels.cpu().data.numpy()
        
        triplets = []
        
        for image_index in range(len(embeddings)):
            # Anchors embeddings are same length as image embeddings
            positive_anchor_index = image_index
            # Negative anchors are those for different brands
            label = labels[image_index]
            same_label_mask = (labels == label)
            negative_indices = np.where(np.logical_not(same_label_mask))[0]
            
            ap_distance = distance_matrix[image_index, positive_anchor_index]
            # Each image has many possible negative anchors.
            # The negative_selection_fn function decides which negative anchor to use. 
            loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([image_index])), torch.LongTensor(negative_indices)] + self.margin
            loss_values = loss_values.data.cpu().numpy()
            hard_negative = self.negative_selection_fn(loss_values)

            if hard_negative is not None:
                hard_negative_anchor_index = negative_indices[hard_negative]
                triplets.append([image_index, positive_anchor_index, hard_negative_anchor_index])
        
        if len(triplets) == 0:
            triplets.append([0, 0, negative_indices[0]])
            
        triplets = np.array(triplets)   
        logger.debug(f"Triplets choosen {triplets}")  
        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return SoccerPersonTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return SoccerPersonTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return SoccerPersonTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)