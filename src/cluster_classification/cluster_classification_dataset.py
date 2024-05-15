import os
import random
import sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from SNGS_dataset.groundtruth_reader import GroundTruthReader
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class BBox:
    frame: int
    x_image: float
    y_image: float
    w_image: float
    h_image: float
    x_pitch: float
    y_pitch: float


@dataclass
class Track:
    track_id: int
    role_id: int
    video: str
    bboxes: List[BBox]
    def __str__(self):
        return f'Track(track_id={self.track_id}, role_id={self.role_id}, video={self.video}, bboxes=(lenght: {len(self.bboxes)}))'
    def __repr__(self):
        return self.__str__()


@dataclass
class ClusterScenario:
    this_tracks_role_id: int
    this_tracks: List[Track]
    rest_tracks: List[Track]
    is_valid: bool
    def __str__(self):
        return f'ClusterScenario(this_tracks_role_id={self.this_tracks_role_id}, this_tracks=(lenght: {len(self.this_tracks)}), rest_tracks=(lenght: {len(self.rest_tracks)}), is_valid={self.is_valid})'
    def __repr__(self):
        return self.__str__()


def get_tracks(videos_list: List[str], gt_reader: GroundTruthReader) -> Dict[str, List[Track]]:
    total_tracks = defaultdict(list)
    for video in tqdm(videos_list, total=len(videos_list), desc='Getting tracks'):
        if video[:5] != "SNGS-":
            continue
        video_tracks = []
        video_annotations = gt_reader.get_annotations_from_video(video)
        for annotation in video_annotations:
            if annotation.track_id not in [track.track_id for track in video_tracks]:
                track = Track(track_id=annotation.track_id, role_id=annotation.role_id, video=annotation.video, bboxes=[])
                video_tracks.append(track)
            bbox = BBox(frame=annotation.image_id,
                        x_image=annotation.bbox["x"],
                        y_image=annotation.bbox["y"],
                        w_image=annotation.bbox["w"],
                        h_image=annotation.bbox["h"],
                        x_pitch=float(round(annotation.bbox["x_pitch"], 4)) if annotation.bbox["x_pitch"] else None, # it is possible for this value to be None
                        y_pitch=float(round(annotation.bbox["y_pitch"], 4)) if annotation.bbox["y_pitch"] else None  # it is possible for this value to be None
                       )
            for track in video_tracks:
                if track.track_id == annotation.track_id:
                    track.bboxes.append(bbox)
        total_tracks[video] = video_tracks
    return total_tracks


class ClusterClassificationDataset(Dataset):
    def __init__(self, data_path: str, videos_list: List[str]=None, 
                 homogeneity_threshold: float=0.9, scenarios_per_video: int=100) -> None:

        self.data_path = data_path
        self.videos_list = videos_list if videos_list else os.listdir(data_path)
        self.homogeneity_threshold = homogeneity_threshold

        # calculate the amount of valid and invalid scenarios per video, given "scenarios_per_video"
        self.amount_of_valid_scenarios = scenarios_per_video // 2
        self.amount_of_invalid_scenarios = scenarios_per_video - self.amount_of_valid_scenarios

        # instantiate GroundTruthReader and get all tracks
        self.gt_reader = GroundTruthReader(self.data_path, self.videos_list)
        self.tracks = get_tracks(self.videos_list, self.gt_reader)
        self.cluster_scenarios = []

        for video in tqdm(self.videos_list, total=len(self.videos_list), desc='Initializing ClusterClassificationDataset'):
            # filter non-videos directories
            if video[:5] != "SNGS-":
                continue

            # get every role_id present in the video
            role_id_in_video = set()
            for track in self.tracks[video]:
                role_id_in_video.add(track.role_id)

            # create valid ClusterScenarios
            for valid_scenario in tqdm(range(self.amount_of_valid_scenarios), total=self.amount_of_valid_scenarios, desc='Generating valid scenarios'):
                # randomly choose a role_id and get the tracks that have it as role_id, and the ones that do not
                random_role = random.choice(list(role_id_in_video))
                random_role_tracks = [track for track in self.tracks[video] if track.role_id == random_role]
                other_role_tracks = [track for track in self.tracks[video] if track.role_id != random_role]

                # within this_tracks, calculate the amount of tracks that will be from the ramdom role
                num_this_tracks_from_random_role = round(len(random_role_tracks) * self.homogeneity_threshold)
                num_this_tracks_from_other_role = len(random_role_tracks) - num_this_tracks_from_random_role
                
                # randomly choose the tracks that will be from the random role and the ones from other roles
                this_tracks_from_random_role = random.sample(random_role_tracks, num_this_tracks_from_random_role)
                this_tracks_from_other_role = random.sample(other_role_tracks, num_this_tracks_from_other_role)

                # concatenate the tracks that will be in this_tracks
                this_tracks = this_tracks_from_random_role + this_tracks_from_other_role

                # the rest of the tracks will be in rest_tracks
                rest_tracks = [track for track in self.tracks[video] if track not in this_tracks]

                # check if the split was done correctly
                assert (this_tracks + rest_tracks).sort == self.tracks[video].sort, f"Error after splitting tracks: this_tracks + rest_tracks != tracks[video]"

                # create the ClusterScenario and append it to the list
                cluster_scenario = ClusterScenario(this_tracks_role_id=random_role, this_tracks=this_tracks, rest_tracks=rest_tracks, is_valid=True)
                self.cluster_scenarios.append(cluster_scenario)

            for invalid_scenario in tqdm(range(self.amount_of_invalid_scenarios), total=self.amount_of_invalid_scenarios, desc='Generating invalid scenarios'):
                # randomly choose a role_id and get the tracks that have it as role_id
                random_role = random.choice(list(role_id_in_video))
                random_role_tracks = [track for track in self.tracks[video] if track.role_id == random_role]

                # get the tracks of other role_id, splitted by role_id
                other_role_tracks = defaultdict(list)
                for track in self.tracks[video]:
                    if track.role_id != random_role:
                        other_role_tracks[track.role_id].append(track)

                # make random choice to form this_tracks

                # the rest of the tracks will be in rest_tracks
                rest_tracks = [track for track in self.tracks[video] if track not in this_tracks]

                # check if the split was done correctly
                assert (this_tracks + rest_tracks).sort == self.tracks[video].sort, f"Error after splitting tracks: this_tracks + rest_tracks != tracks[video]"

                # create the ClusterScenario and append it to the list
                cluster_scenario = ClusterScenario(this_tracks_role_id=random_role, this_tracks=this_tracks, rest_tracks=rest_tracks, is_valid=False)
                self.cluster_scenarios.append(cluster_scenario)

    def __len__(self) -> int:
        return len(self.cluster_scenarios)

    def __getitem__(self, idx: int) -> ClusterScenario:
        return self.cluster_scenarios[idx], self.cluster_scenarios[idx].this_tracks_role_id


if __name__ == "__main__":
    data_path = "../../data/SoccerNetGS/valid"
    videos_list = ["SNGS-021", "SNGS-039", "SNGS-078"]
    homogeneity_threshold = 0.9
    scenarios_per_video = 100

    dataset = ClusterClassificationDataset(data_path, videos_list, homogeneity_threshold, scenarios_per_video)

    for i in range(len(dataset)):
        scenario, role_id = dataset[i]
        print(scenario)
        print(role_id)
