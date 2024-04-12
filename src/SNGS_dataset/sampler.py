import numpy as np
from typing import Any
from torch.utils.data import BatchSampler


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

    def __iter__(self) -> Any:
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            videos = np.random.choice(list(self.videos), self.n_videos, replace=False)
            indices = []
            for video in videos:
                indices.extend(
                    self.labels_indices_from_videos[video][
                        self.used_label_indices_count[
                            video
                        ] : self.used_label_indices_count[video]
                        + self.n_samples
                    ]
                )
                self.used_label_indices_count[video] += self.n_samples
                if self.used_label_indices_count[video] + self.n_samples > len(
                    self.labels_indices_from_videos[video]
                ):
                    np.random.shuffle(self.labels_indices_from_videos[video])
                    self.used_label_indices_count[video] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self) -> int:
        return self.n_dataset // self.batch_size


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import sys

    sys.path.append("../models")
    from dataset import SoccerNetGSDataset
    from models.resnet import TRANSFORM

    dataset = SoccerNetGSDataset("../../data/SoccerNetGS/train", transform=TRANSFORM)
    sampler = SoccerNetGSBatchSampler(dataset.labels, 1, 30)
    loader = DataLoader(dataset=dataset, batch_sampler=sampler)

    print(f"Number of batches: {len(sampler)}")

    # batch_idx_limit = 5
    for batch_idx, data in enumerate(loader):
    # for batch_idx, (images, labels) in enumerate(loader):
        
        # if batch_idx == batch_idx_limit:
        #     break
        print(data)
        # print(f"model input size: {images.shape}")
        # print(f"labels size: {labels['role_id'].shape}")
        print("//////////////////////////")
