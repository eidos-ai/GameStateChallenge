"""Module to render a random frame from a SoccerNetGS
video with object annotations and pitch lines."""

from typing import Any
import json
import random
import os
import sys
import argparse
import cv2 as cv  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt


class SoccerNetGS:
    """Class to render a random frame from a SoccerNetGS
    video with object annotations and pitch lines."""

    DATA_DIRECTORY = "./data/SoccerNetGS/gamestate-2024/"
    RADAR_PATH = "./sn-gamestate/sn_gamestate/visualization/Radar.png"

    def __init__(self) -> None:
        self.palette = {  # RGB values
            "Big rect. left bottom": (127, 0, 0),
            "Big rect. left main": (102, 102, 102),
            "Big rect. left top": (0, 0, 127),
            "Big rect. right bottom": (86, 32, 39),
            "Big rect. right main": (48, 77, 0),
            "Big rect. right top": (14, 97, 100),
            "Circle central": (0, 0, 255),
            "Circle left": (255, 127, 0),
            "Circle right": (0, 255, 255),
            "Goal left crossbar": (255, 255, 200),
            "Goal left post left ": (165, 255, 0),
            "Goal left post left": (165, 255, 0),
            "Goal left post right": (155, 119, 45),
            "Goal right crossbar": (86, 32, 139),
            "Goal right post left": (196, 120, 153),
            "Goal right post right": (166, 36, 52),
            "Goal unknown": (0, 0, 0),
            "Line unknown": (0, 0, 0),
            "Middle line": (255, 255, 0),
            "Side line bottom": (255, 0, 255),
            "Side line left": (0, 255, 150),
            "Side line right": (0, 230, 0),
            "Side line top": (230, 0, 0),
            "Small rect. left bottom": (0, 150, 255),
            "Small rect. left main": (254, 173, 225),
            "Small rect. left top": (87, 72, 39),
            "Small rect. right bottom": (122, 0, 255),
            "Small rect. right main": (255, 255, 255),
            "Small rect. right top": (153, 23, 153),
        }

    def find_video_path(self, video_number: str) -> tuple[str, str, str]:
        """Find the path of the video and the frame to be displayed."""
        video_directory = "SNGS-" + str(video_number).zfill(3)
        video_path = ""
        for root, dirs, _ in os.walk(self.DATA_DIRECTORY):
            if video_directory in dirs:
                if "challenge" not in str(os.path.join(root, video_directory)):
                    video_path = os.path.join(root, video_directory)
                else:
                    print(
                        f"Video {video_directory} is a challenge video. Try another video number."
                    )
                    sys.exit()
        if video_path == "":
            print(f"Video {video_directory} not found.")
            sys.exit()
        else:
            frame_directory = video_path + "/img1/"
        random_frame = str(random.randint(0, 750)).zfill(
            6
        )  # 3 digit integer number of the frame
        frame_name = (
            str(random_frame).zfill(6) + ".jpg"
        )  # filled 6 digit frame file name
        frame_path = frame_directory + frame_name

        print(f"The frame {frame_name} from video {video_directory} will be displayed.")
        return video_path, random_frame, frame_path

    def get_object_annotations(
        self, video_path: str, random_frame: str
    ) -> list[dict[str, Any]]:
        """Find the object annotations for the random frame."""
        object_annotations = []
        object_categories = ["player", "goalkeeper", "referee"]

        with open(video_path + "/Labels-GameState.json", "r", encoding="utf-8") as file:
            data = json.load(file)
            for annotation in data["annotations"]:
                try:
                    if (
                        annotation["image_id"][-6:] == random_frame
                        and annotation["attributes"]["role"] in object_categories
                    ):
                        object_annotations.append(annotation)
                except KeyError:
                    pass

        return object_annotations

    def get_pitch_annotation(
        self, video_path: str, random_frame: str
    ) -> dict[str, Any]:
        """Find the pitch annotations for the random frame."""
        pitch_annotation = {}
        with open(video_path + "/Labels-GameState.json", "r", encoding="utf-8") as file:
            data = json.load(file)
            for annotation in data["annotations"]:
                try:
                    if (
                        annotation["image_id"][-6:] == random_frame
                        and annotation["supercategory"] == "pitch"
                    ):
                        pitch_annotation = annotation
                except KeyError:
                    pass
        return pitch_annotation

    def render_objects(
        self, frame: np.ndarray[Any, np.dtype[Any]], object_annotations: list[Any]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Render the objects labelling."""
        for item in object_annotations:
            # object rectangle
            starting_point_object = (
                (int(item["bbox_image"]["x"])),
                int(item["bbox_image"]["y"]),
            )
            ending_point_object = (
                int(item["bbox_image"]["x"]) + int(item["bbox_image"]["w"]),
                int(item["bbox_image"]["y"]) + int(item["bbox_image"]["h"]),
            )

            # color rectangle depending on team (if player or goalkeeper)
            if (
                item["attributes"]["role"] == "player"
                or item["attributes"]["role"] == "goalkeeper"
            ):
                if item["attributes"]["team"] == "left":
                    color = (0, 0, 255)  # Red according to opencv
                else:
                    color = (255, 0, 0)  # Blue according to opencv
                # jersey number rectangle
                starting_point_jersey = (
                    starting_point_object[0] - 15,
                    ending_point_object[1],
                )
                ending_point_jersey = (
                    ending_point_object[0] + 15,
                    starting_point_jersey[1] + 25,
                )
                cv.rectangle(
                    frame, starting_point_jersey, ending_point_jersey, color, -1
                )
                # jersey number rectangle text
                jersey_text = "JN: "
                if item["attributes"]["jersey"] is not None:
                    jersey_text += item["attributes"]["jersey"]
                else:
                    jersey_text += "-"
                cv.putText(
                    frame,
                    jersey_text,
                    (starting_point_jersey[0] + 5, starting_point_jersey[1] + 15),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

            # color rectangle (if referee)
            elif item["attributes"]["role"] == "referee":
                color = (30, 255, 255)

            thickness = 2  # Line thickness of 2px
            cv.rectangle(
                frame, starting_point_object, ending_point_object, color, thickness
            )
        return frame

    def render_minimap(
        self, frame: np.ndarray[Any, np.dtype[Any]], object_annotations: list[Any]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Render the minimap."""
        minimap = cv.imread(self.RADAR_PATH)
        for item in object_annotations:
            if (
                item["attributes"]["role"] == "player"
                or item["attributes"]["role"] == "goalkeeper"
            ):
                if item["attributes"]["team"] == "left":
                    color = (0, 0, 255)  # Red according to opencv
                else:
                    color = (255, 0, 0)  # Blue according to opencv
                try:
                    starting_point_minimap = (
                        (int(item["bbox_pitch"]["x_bottom_middle"])) + 61,
                        (int(item["bbox_pitch"]["y_bottom_middle"])) + 37,
                    )
                    ending_point_minimap = (
                        (int(item["bbox_pitch"]["x_bottom_middle"])) + 63,
                        (int(item["bbox_pitch"]["y_bottom_middle"])) + 39,
                    )
                    cv.rectangle(
                        minimap, starting_point_minimap, ending_point_minimap, color, -1
                    )
                    resized_minimap = cv.resize(
                        minimap,
                        (int(minimap.shape[1] * 3), int(minimap.shape[0] * 3)),
                        interpolation=cv.INTER_AREA,
                    )
                except TypeError:
                    continue
        frame[
            800 : resized_minimap.shape[0] + 800, 750 : resized_minimap.shape[1] + 750
        ] = resized_minimap
        return frame

    def render_pitch_lines(
        self, frame: np.ndarray[Any, np.dtype[Any]], pitch_annotation: dict[str, Any]
    ) -> np.ndarray[Any, np.dtype[Any]]:
        """Render the pitch lines."""
        for line in pitch_annotation["lines"]:
            points = np.array([])
            for point in pitch_annotation["lines"][line]:
                coords = (int(float(point["x"]) * 1920), int(float(point["y"]) * 1080))
                points = np.append(points, coords, axis=0)
            ctr = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
            cv.drawContours(frame, [ctr], -1, self.palette[line], 2)
        return frame

    def show_image(
        self,
        frame_path: str,
        object_annotations: list[Any],
        pitch_annotation: dict[str, Any],
        objects_render: bool = True,
        minimap_render: bool = True,
        pitch_lines_render: bool = True,
    ) -> None:
        """Show the image."""
        frame = cv.imread(frame_path)
        if pitch_lines_render:
            frame = self.render_pitch_lines(frame, pitch_annotation)
        if objects_render:
            frame = self.render_objects(frame, object_annotations)
        if minimap_render:
            frame = self.render_minimap(frame, object_annotations)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.show()


if __name__ == "__main__":
    sn = SoccerNetGS()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-v", "--video_number", type=int, required=True, help="Choose video number."
    )
    args = parser.parse_args()
    path_video, frame_random, path_frame = sn.find_video_path(args.video_number)
    annotations_objects = sn.get_object_annotations(path_video, frame_random)
    annotation_pitch = sn.get_pitch_annotation(path_video, frame_random)
    sn.show_image(path_frame, annotations_objects, annotation_pitch)
