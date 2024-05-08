import math
import numpy as np
from typing import Any, List
from tqdm import tqdm
from operator import itemgetter
from torch.utils.data import BatchSampler

class SoccerNetEvenRoleIdBatchSampler(BatchSampler):
    """
    This batch sampler returns a balanced number of samples among roles_ids for a video.
    When a role_id runs out of instances for a match, it starts again from start.
    """
    def __init__(
        self, labels: list[dict[str:Any]], n_videos: int, n_samples: int
    ) -> None:
        self.labels = labels
        self.n_videos = n_videos
        self.n_samples = n_samples
        self.batch_size = self.n_videos * self.n_samples
        self.n_dataset = len(self.labels)

        # { video_name: { roles_id: list[indices_of_crops_from_that_video_and_role] } }
        self.labels_indices_from_videos = {}
        for idx, label in enumerate(self.labels):
            video = label["video"]
            roles_id = label["role_id"]
            if video not in self.labels_indices_from_videos:
                self.labels_indices_from_videos[video] = {}
            if roles_id not in self.labels_indices_from_videos[video]:
                self.labels_indices_from_videos[video][roles_id] = []
            self.labels_indices_from_videos[video][roles_id].append(idx)
        
        # Shuffle indices for each roles_id in each video
        for video in self.labels_indices_from_videos:
            for roles_id in self.labels_indices_from_videos[video]:
                np.random.shuffle(self.labels_indices_from_videos[video][roles_id])

        self.used_label_indices_count = {
            video: {roles_id: 0 for roles_id in self.labels_indices_from_videos[video]}
            for video in self.labels_indices_from_videos
        }

        self.count = 0   
    
    def __iter__(self) -> List[int]:
        self.count = 0
        while self.count < self.n_dataset:
            remaining_samples = self.n_dataset - self.count
            current_batch_size = min(self.batch_size, remaining_samples)

            if remaining_samples < 2 * self.batch_size:
                current_batch_size = remaining_samples
                
            indices = []
            while len(indices) < current_batch_size:
                videos = np.random.choice(list(self.labels_indices_from_videos.keys()), self.n_videos, replace=False)
                
                for video in videos:
                    roles_in_video = len(self.labels_indices_from_videos[video])
                    for roles_id in self.labels_indices_from_videos[video]:
                        available_samples = len(self.labels_indices_from_videos[video][roles_id]) - self.used_label_indices_count[video][roles_id]
                        samples_needed = min(self.n_samples // roles_in_video, available_samples)
                        
                        start_index = self.used_label_indices_count[video][roles_id]
                        end_index = start_index + samples_needed
                        indices.extend(self.labels_indices_from_videos[video][roles_id][start_index:end_index])

                        # Wrap around if the end of the list is reached
                        if end_index >= len(self.labels_indices_from_videos[video][roles_id]):
                            end_index = end_index % len(self.labels_indices_from_videos[video][roles_id])
                            indices.extend(self.labels_indices_from_videos[video][roles_id][:end_index])

                        # Update count of used samples
                        self.used_label_indices_count[video][roles_id] = end_index
                    
            self.count += len(indices)
            
            videos = np.unique([self.labels[i]['video'] for i in indices])
            # for v in videos:
            #     print("/////")
            #     print(v)
            #     vi = [i for i in indices if self.labels[i]['video'] == v]
            #     roles_ids = np.unique([self.labels[i]['role_id'] for i in vi])
            #     for r in roles_ids:
            #         print("Role", r, len([i for i in vi if self.labels[i]["role_id"] == r]))                
            yield indices
        
        # Once finished, reshuffle and reset count for every roles_id in every video
        for video in self.labels_indices_from_videos:
            for roles_id in self.labels_indices_from_videos[video]:
                np.random.shuffle(self.labels_indices_from_videos[video][roles_id])
                self.used_label_indices_count[video][roles_id] = 0
            
    def __len__(self) -> int:
        size = 0
        for video in self.labels_indices_from_videos:
            for roles_id in self.labels_indices_from_videos[video]:
                size += math.floor(len(self.labels_indices_from_videos[video][roles_id]) / self.batch_size)
        return size
    
class SoccerNetGSBatchSampler(BatchSampler):
    """
    BatchSampler samples from n_videos and within them, samples n_samples.
    Returns batches of size n_videos * n_samples
    """

    def __init__(
        self, labels: list[dict[str:Any]], n_videos: int, n_samples: int
    ) -> None:
        self.labels = labels
        self.n_videos = n_videos
        self.n_samples = n_samples
        self.batch_size = self.n_videos * self.n_samples
        self.n_dataset = len(self.labels)

        # { video_name: list[indices_of_crops_from_that_video] }
        self.labels_indices_from_videos = {}
        for idx, label in enumerate(self.labels):
            video = label["video"]
            if video not in self.labels_indices_from_videos:
                self.labels_indices_from_videos[video] = []
            self.labels_indices_from_videos[video].append(idx)
        # shuffle indices crops in each video
        for video in self.labels_indices_from_videos:
            np.random.shuffle(self.labels_indices_from_videos[video])

        # set with all the videos
        self.videos = set(self.labels_indices_from_videos.keys())

        # { video: count_of_samples_used_from_that_video }
        self.used_label_indices_count = {video: 0 for video in self.videos}
        self.count = 0

    def __iter__(self) -> List[int]:
        self.count = 0
        while self.count < self.n_dataset:
            # Calculate the remaining samples and decide the size of the next batch
            remaining_samples = self.n_dataset - self.count
            current_batch_size = min(self.batch_size, remaining_samples)
            
            # If there are less than two batches of samples remaining
            if remaining_samples < 2*self.batch_size: # and self.n_dataset % self.batch_size == 0
                current_batch_size = remaining_samples  # Make the last batch larger by adding the remainder

            # Try to collect the required number of samples from each video
            indices = []
            while len(indices) < current_batch_size:
                # Sample n_videos that still have annotations left
                videos = np.random.choice(list(self.videos), self.n_videos, replace=False)
                for video in videos:
                    available_samples = len(self.labels_indices_from_videos[video]) - self.used_label_indices_count[video]
                    samples_needed = min(self.n_samples, available_samples)

                    # Collect samples
                    start_index = self.used_label_indices_count[video]
                    end_index = start_index + samples_needed
                    indices.extend(self.labels_indices_from_videos[video][start_index:end_index])

                    # Update count of used samples
                    self.used_label_indices_count[video] += samples_needed
                    
            self.count += len(indices)
            yield indices
            
        # reshuffle and reset count for every video
        for video in self.videos:
            np.random.shuffle(self.labels_indices_from_videos[video])
            self.used_label_indices_count[video] = 0

    def __len__(self) -> int:
        size = 0
        for video in self.labels_indices_from_videos.keys():
            size += math.floor(len(self.labels_indices_from_videos[video]) / self.batch_size)
        # return math.floor(self.n_dataset / self.batch_size)
        return size

def convert_dict_to_list(data):
    """
    Convert a complex dictionary with lists and tensors into a list of dictionaries where each dictionary contains
    individual elements from the lists and tensors.

    Args:
    data (dict): A dictionary containing lists and tensors.

    Returns:
    list: A list of dictionaries with individual elements.
    """
    list_of_dicts = []
    for i in range(len(data['annotation_id'])):
        entry = {
            'annotation_id': data['annotation_id'][i],
            'video': data['video'][i],
            'image_id': data['image_id'][i],
            'track_id': data['track_id'][i].item(),
            'role': data['role'][i],
            'team': data['team'][i],
            'role_id': data['role_id'][i].item(),
            'bbox': {
                'x': data['bbox']['x'][i].item(),
                'y': data['bbox']['y'][i].item(),
                'w': data['bbox']['w'][i].item(),
                'h': data['bbox']['h'][i].item()
            }
        }
        list_of_dicts.append(entry)
    return list_of_dicts

def compare_lists_of_dicts(list1, list2):
    """
    Compares two lists of dictionaries to check if they contain the same items, irrespective of the order of the items.

    Args:
    list1 (list): The first list of dictionaries.
    list2 (list): The second list of dictionaries.

    Returns:
    bool: True if the lists contain the same dictionaries, False otherwise.
    """
    from collections import Counter
    import json

    # Convert the dictionaries to strings to make them hashable and comparable
    list1_str = [json.dumps(d, sort_keys=True) for d in list1]
    list2_str = [json.dumps(d, sort_keys=True) for d in list2]

    # Use Counter to compare the lists of strings
    return Counter(list1_str) == Counter(list2_str)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import sys

    sys.path.append("../models")
    from dataset import SoccerNetGSDataset
    from models.resnet import TRANSFORM

    dataset = SoccerNetGSDataset("../../data/SoccerNetGS/train", transform=TRANSFORM, videos_list=["SNGS-060", "SNGS-097", "SNGS-103"])
    sampler = SoccerNetGSBatchSampler(dataset.labels, 3, 66)
    loader = DataLoader(dataset=dataset, batch_sampler=sampler, num_workers=0)

    print(f"Dataset length: {len(dataset)}")
    print(f"Number of batches: {len(sampler)}")
    print(f"Batch size: {sampler.batch_size}")

    batch_idx_limit = -1
    returned_labels = []
    for batch_idx, (images, labels) in tqdm(enumerate(loader), total=len(sampler)):
        if batch_idx == batch_idx_limit:
            break
        if len(labels["annotation_id"]) != sampler.batch_size:
            print(len(labels["annotation_id"]))
        returned_labels.extend(convert_dict_to_list(labels))

    assert compare_lists_of_dicts(dataset.labels, returned_labels), "Returned labels don't match with dataset labels"
    assert batch_idx + 1 == len(sampler), f"Incorrect number of returned batches, expected {len(sampler)} and got {batch_idx + 1}"
