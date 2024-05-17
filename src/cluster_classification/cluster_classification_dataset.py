import os
import random
import sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from SNGS_dataset.groundtruth_reader import GroundTruthReader
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import tqdm


class SkipIteration(Exception):
    pass


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
    this_tracks_role_id: Optional[int]
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
        self.videos_list = videos_list if videos_list else [video for video in os.listdir(data_path) if video[:5] == "SNGS-"]
        self.homogeneity_threshold = homogeneity_threshold

        # calculate the amount of valid and invalid scenarios per video, given "scenarios_per_video"
        amount_of_valid_scenarios = scenarios_per_video // 2
        amount_of_invalid_scenarios = scenarios_per_video - amount_of_valid_scenarios

        # instantiate GroundTruthReader and get all tracks
        gt_reader = GroundTruthReader(self.data_path, self.videos_list)
        self.tracks = get_tracks(self.videos_list, gt_reader)
        self.cluster_scenarios = []

        # iterate over all videos
        for video in tqdm(self.videos_list, total=len(self.videos_list), desc='Initializing ClusterClassificationDataset'):

            # get every role_id present in the video
            role_id_in_video = {track.role_id for track in self.tracks[video]}

            # create valid ClusterScenarios
            for valid_scenario in tqdm(range(amount_of_valid_scenarios), total=amount_of_valid_scenarios, desc=f'Valid scenarios of video {video}'):
                # randomly choose a role_id and get the tracks that have it as role_id, and the ones that do not
                random_role = random.choice(list(role_id_in_video))
                random_role_tracks = [track for track in self.tracks[video] if track.role_id == random_role]
                other_role_tracks = [track for track in self.tracks[video] if track.role_id != random_role]

                # within this_tracks, calculate the amount of tracks that will be from the ramdom role
                num_this_tracks_from_random_role = round(len(random_role_tracks) * self.homogeneity_threshold)
                num_this_tracks_from_other_role = len(random_role_tracks) - num_this_tracks_from_random_role
                
                # randomly choose the tracks that will be from the random role and the ones from other roles
                this_tracks = random.sample(random_role_tracks, num_this_tracks_from_random_role)
                this_tracks.extend(random.sample(other_role_tracks, num_this_tracks_from_other_role))

                # the rest of the tracks will be in rest_tracks
                rest_tracks = [track for track in self.tracks[video] if track not in this_tracks]

                # check if the split was done correctly
                sorted_list = sorted((this_tracks + rest_tracks), key=lambda track: track.track_id)
                assert sorted_list == self.tracks[video], f"Error after splitting tracks: this_tracks + rest_tracks != tracks[video]"

                # create the ClusterScenario and append it to the list
                cluster_scenario = ClusterScenario(this_tracks_role_id=random_role, this_tracks=this_tracks, rest_tracks=rest_tracks, is_valid=True)
                self.cluster_scenarios.append(cluster_scenario)

            for invalid_scenario in tqdm(range(amount_of_invalid_scenarios), total=amount_of_invalid_scenarios, desc=f'Invalid scenarios of video {video}'):
                # randomly choose a role_id and get the tracks that have it as role_id, and the ones that do not
                try:
                    random_role = random.choice(list(role_id_in_video))
                    random_role_tracks = [track for track in self.tracks[video] if track.role_id == random_role]
                    other_role_tracks = [track for track in self.tracks[video] if track.role_id != random_role]

                    # make random choice to form this_tracks
                    decision = random.choice([1, 2, 3])

                    # DECISION: Tomar 50% de tracks de ese role_id + Tomar 50% de tracks de otro role_id
                    if decision == 1:
                        # get 50% of tracks from random_role
                        num_this_tracks_from_random_role = round(len(random_role_tracks) // 2)                    
                        this_tracks = random.sample(random_role_tracks, num_this_tracks_from_random_role)

                        # get 50% of tracks from other_random_role
                        min_num_this_tracks_from_other_random_role = round(num_this_tracks_from_random_role * 0.7)
                        max_num_this_tracks_from_other_random_role = round(num_this_tracks_from_random_role * 1.3)
                        other_random_role = random.choice(list(role_id_in_video))

                        counter = 0
                        while (other_random_role == random_role or 
                            len([track for track in self.tracks[video] if track.role_id == other_random_role]) < min_num_this_tracks_from_other_random_role):
                            other_random_role = random.choice(list(role_id_in_video))
                            counter += 1
                            if counter == 10:
                                raise SkipIteration
                        other_random_role_tracks = [track for track in self.tracks[video] if track.role_id == other_random_role]

                        if len(other_random_role_tracks) < max_num_this_tracks_from_other_random_role:
                            this_tracks.extend(other_random_role_tracks)
                        else:
                            this_tracks.extend(random.sample(other_random_role_tracks, min_num_this_tracks_from_other_random_role))

                    # DECISION: Tomar 100% de tracks de ese role_id + Tomar random 50% de tracks de otro role_id
                    elif decision == 2:
                        # get 100% of tracks from random_role
                        num_this_tracks_from_random_role = len(random_role_tracks)                    
                        this_tracks = random.sample(random_role_tracks, num_this_tracks_from_random_role)

                        # get 50% of tracks from other_random_role
                        min_num_this_tracks_from_other_random_role = round((num_this_tracks_from_random_role / 2) * 0.7)
                        max_num_this_tracks_from_other_random_role = round((num_this_tracks_from_random_role / 2) * 1.3)
                        other_random_role = random.choice(list(role_id_in_video))

                        counter = 0
                        while (other_random_role == random_role or 
                            len([track for track in self.tracks[video] if track.role_id == other_random_role]) < min_num_this_tracks_from_other_random_role):
                            other_random_role = random.choice(list(role_id_in_video))
                            counter += 1
                            if counter == 10:
                                raise SkipIteration
                        other_random_role_tracks = [track for track in self.tracks[video] if track.role_id == other_random_role]

                        if len(other_random_role_tracks) < max_num_this_tracks_from_other_random_role:
                            this_tracks.extend(other_random_role_tracks)
                        else:
                            this_tracks.extend(random.sample(other_random_role_tracks, min_num_this_tracks_from_other_random_role))

                    # DECISION: Tomar 100% de tracks de ese role_id + Tomar random 100% de tracks de otro role_id
                    else:
                        # get 100% of tracks from random_role
                        num_this_tracks_from_random_role = len(random_role_tracks)                    
                        this_tracks = random.sample(random_role_tracks, num_this_tracks_from_random_role)

                        # get 100% of tracks from other_random_role
                        min_num_this_tracks_from_other_random_role = round((num_this_tracks_from_random_role) * 0.7)
                        max_num_this_tracks_from_other_random_role = round((num_this_tracks_from_random_role) * 1.3)
                        other_random_role = random.choice(list(role_id_in_video))

                        counter = 0
                        while (other_random_role == random_role or 
                            len([track for track in self.tracks[video] if track.role_id == other_random_role]) < min_num_this_tracks_from_other_random_role):
                            other_random_role = random.choice(list(role_id_in_video))
                            counter += 1
                            if counter == 10:
                                raise SkipIteration
                        other_random_role_tracks = [track for track in self.tracks[video] if track.role_id == other_random_role]

                        if len(other_random_role_tracks) < max_num_this_tracks_from_other_random_role:
                            this_tracks.extend(other_random_role_tracks)
                        else:
                            this_tracks.extend(random.sample(other_random_role_tracks, min_num_this_tracks_from_other_random_role))

                    # the rest of the tracks will be in rest_tracks
                    rest_tracks = [track for track in self.tracks[video] if track not in this_tracks]

                    # check if the split was done correctly
                    sorted_list = sorted((this_tracks + rest_tracks), key=lambda track: track.track_id)
                    assert sorted_list == self.tracks[video], f"Error after splitting tracks: this_tracks + rest_tracks != tracks[video]"

                    # create the ClusterScenario and append it to the list
                    cluster_scenario = ClusterScenario(this_tracks_role_id=None, this_tracks=this_tracks, rest_tracks=rest_tracks, is_valid=False)
                    self.cluster_scenarios.append(cluster_scenario)
                
                except SkipIteration:
                    continue

    def __len__(self) -> int:
        return len(self.cluster_scenarios)

    def __getitem__(self, idx: int) -> ClusterScenario:
        return self.cluster_scenarios[idx]


if __name__ == "__main__":
    train_data_path = "../../data/SoccerNetGS/train"
    valid_data_path = "../../data/SoccerNetGS/valid"

    train_dataset = ClusterClassificationDataset(train_data_path)
    valid_dataset = ClusterClassificationDataset(valid_data_path)
