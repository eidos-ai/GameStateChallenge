import torch.nn as nn
import torch.nn.functional as F

def distance(tensor_a, tensor_b):
    return (tensor_a - tensor_b).pow(2).sum(1)

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, anchors_embeddings, target):
        
        triplets = self.triplet_selector.get_triplets(embeddings, anchors_embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        
        # Triplets are given as image_index, anchor_positive_index, anchor_negative_index.
        # So last indexes are taken from anchors_embeddings
        ap_distances = distance(embeddings[triplets[:, 0]], anchors_embeddings[triplets[:, 1]])
        an_distances = distance(embeddings[triplets[:, 0]], anchors_embeddings[triplets[:, 2]])
        losses = F.relu(ap_distances - an_distances + self.margin)
        
        ap_distances_mean = ap_distances.mean().item()
        an_distances_mean = an_distances.mean().item()
        return losses.mean(), len(triplets), ap_distances_mean, an_distances_mean
