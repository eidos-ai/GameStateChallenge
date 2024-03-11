import cv2 as cv
import json
import random
import os
import argparse
import numpy as np

class SoccerNetGS:
    def __init__(self):
        self.data_directory = "./data/SoccerNetGS/gamestate-2024/"
        self.radar_path = "./sn-gamestate/sn_gamestate/visualization/Radar.png"
        self.palette = { # RGB values
            'Big rect. left bottom': (127, 0, 0),
            'Big rect. left main': (102, 102, 102),
            'Big rect. left top': (0, 0, 127),
            'Big rect. right bottom': (86, 32, 39),
            'Big rect. right main': (48, 77, 0),
            'Big rect. right top': (14, 97, 100),
            'Circle central': (0, 0, 255),
            'Circle left': (255, 127, 0),
            'Circle right': (0, 255, 255),
            'Goal left crossbar': (255, 255, 200),
            'Goal left post left ': (165, 255, 0),
            'Goal left post left': (165, 255, 0),
            'Goal left post right': (155, 119, 45),
            'Goal right crossbar': (86, 32, 139),
            'Goal right post left': (196, 120, 153),
            'Goal right post right': (166, 36, 52),
            'Goal unknown': (0, 0, 0),
            'Line unknown': (0, 0, 0),
            'Middle line': (255, 255, 0),
            'Side line bottom': (255, 0, 255),
            'Side line left': (0, 255, 150),
            'Side line right': (0, 230, 0),
            'Side line top': (230, 0, 0),
            'Small rect. left bottom': (0, 150, 255),
            'Small rect. left main': (254, 173, 225),
            'Small rect. left top': (87, 72, 39),
            'Small rect. right bottom': (122, 0, 255),
            'Small rect. right main': (255, 255, 255),
            'Small rect. right top': (153, 23, 153)
        }

    def find_video_path(self, video_number):

        video_directory = "SNGS-" + str(video_number).zfill(3)

        for root, dirs, files in os.walk(self.data_directory):
            if video_directory in dirs:
                if "challenge" not in str(os.path.join(root, video_directory)):
                    print(os.path.join(root, video_directory))
                    video_path =  os.path.join(root, video_directory)
                else:
                    print(f"Video {video_directory} is a challenge video. Skipping.")
                    exit()

        try:
            frame_directory = video_path + "/img1/"
        except:
            print(f"Video {video_directory} not found.")
            exit()

        random_frame = str(random.randint(0, 750)).zfill(6) # 3 digit integer number of the frame
        frame_name = str(random_frame).zfill(6) + ".jpg" # filled 6 digit frame file name
        frame_path = frame_directory + frame_name

        print(f"The frame {frame_name} from video {video_directory} will be displayed.")

        return video_path, random_frame, frame_path

    def get_object_annotations(self, video_path, random_frame):
        object_annotations = []
        object_categories = ["player", "goalkeeper", "referee"]

        with open(video_path + "/Labels-GameState.json", 'r') as file:
            data = json.load(file)
            for annotation in data["annotations"]:
                try:
                    if annotation["image_id"][-6:] == random_frame and annotation["attributes"]["role"] in object_categories:
                        object_annotations.append(annotation)
                except:
                    pass

        return object_annotations
    
    def get_pitch_annotation(self, video_path, random_frame):

        with open(video_path + "/Labels-GameState.json", 'r') as file:
            data = json.load(file)
            for annotation in data["annotations"]:
                try:
                    if annotation["image_id"][-6:] == random_frame and annotation["supercategory"] == "pitch":
                        return annotation
                except:
                    pass
    
    def render_frame(self, frame_path, object_annotations, pitch_annotation):
        frame = cv.imread(frame_path)
        minimap = cv.imread(self.radar_path)

        # pitch lines rendering
        for line in pitch_annotation["lines"]:
            points = np.array([])
            for point in pitch_annotation["lines"][line]:
                coords = (int(float(point["x"]) * 1920), int(float(point["y"]) * 1080))
                points = np.append(points, coords, axis=0)
            ctr = np.array(points).reshape((-1,1,2)).astype(np.int32)
            cv.drawContours(frame, [ctr], -1, self.palette[line], 2)

        # object (players, goalkeepers, referees) rendering
        for object in object_annotations:

            # object box
            starting_point_object = ((int(object["bbox_image"]["x"])), int(object["bbox_image"]["y"]))
            ending_point_object = (int(object["bbox_image"]["x"]) + int(object["bbox_image"]["w"]), int(object["bbox_image"]["y"]) + int(object["bbox_image"]["h"]))

            # color box depending on team
            if object["attributes"]["role"] == "player" or object["attributes"]["role"] == "goalkeeper":
                if object["attributes"]["team"] == "left":
                    color = (0, 0, 255) # Red according to opencv
                else:
                    color = (255, 0, 0) # Blue according to opencv

                # jersey box
                starting_point_jersey = (starting_point_object[0] - 15 , ending_point_object[1])
                ending_point_jersey = (ending_point_object[0] + 15, starting_point_jersey[1] + 25)
                cv.rectangle(frame, starting_point_jersey, ending_point_jersey, color, -1)

                # jersey text
                jersey_text = "JN: "
                if object["attributes"]["jersey"] != None:
                    jersey_text += object["attributes"]["jersey"]
                else:
                    jersey_text += "-"
                cv.putText(frame, jersey_text, (starting_point_jersey[0] + 5, starting_point_jersey[1] + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

                # player on minimap
                starting_point_minimap = ((int(object["bbox_pitch"]["x_bottom_middle"])) + 61, (int(object["bbox_pitch"]["y_bottom_middle"])) + 37)
                ending_point_minimap = ((int(object["bbox_pitch"]["x_bottom_middle"])) + 63, (int(object["bbox_pitch"]["y_bottom_middle"])) + 39)
                cv.rectangle(minimap, starting_point_minimap, ending_point_minimap, color, -1)
                resized_minimap = cv.resize(minimap, (int(minimap.shape[1] * 3), int(minimap.shape[0] * 3)), interpolation = cv.INTER_AREA)
                
            elif object["attributes"]["role"] == "referee":
                color = (30, 255, 255)

            thickness = 2 # Line thickness of 2 px
            cv.rectangle(frame, starting_point_object, ending_point_object, color, thickness)
                
        frame[800:resized_minimap.shape[0] + 800, 750:resized_minimap.shape[1] + 750] = resized_minimap
        cv.imshow("Random frame", frame)
        cv.waitKey(0) # Wait for a keystroke in the window

if __name__ == "__main__":
    sn = SoccerNetGS()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-v", "--video_number", type=int, required=True, help='Choose video number.')
    args = parser.parse_args()
    video_path, random_frame, frame_path = sn.find_video_path(args.video_number)
    object_annotations = sn.get_object_annotations(video_path, random_frame)
    pitch_annotation = sn.get_pitch_annotation(video_path, random_frame)
    sn.render_frame(frame_path, object_annotations, pitch_annotation)