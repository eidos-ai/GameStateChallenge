import os
import cv2
from tqdm import tqdm
from typing import List
from ultralytics import YOLO

def extract_persons_crops_from_videos(model_path: str, data_path: str, output_folder: str, videos_list: List[str]) -> None:
    model = YOLO(model_path)

    os.makedirs(output_folder, exist_ok=True)
    if videos_list is None:
        videos_list = os.listdir(data_path)
        videos_list = [item for item in videos_list if item[:5] == "SNGS-"]

    for video in tqdm(videos_list, total=len(videos_list)):
        output_video_folder = os.path.join(output_folder, video)
        os.makedirs(output_video_folder, exist_ok=True)
        frames_path = os.path.join(data_path, f"{video}/img1")
        for frame in tqdm(os.listdir(frames_path), total=len(os.listdir(frames_path)), desc=f"Frames of video {video}"):
            count = 0
            img_path = os.path.join(frames_path, frame)
            img = cv2.imread(img_path)
            results = model.predict(img, verbose=False)
            for item in results[0].boxes:
                class_id = int(item.cls[0].item())
                if class_id != 0:
                    continue
                box = item.xywh[0].tolist()
                box = [int(round(num)) for num in box]
                try:
                    crop_img = img[box[1]-int(round(box[3]/2)):box[1]+int(round(box[3]/2)), box[0]-int(round(box[2]/2)):box[0]+int(round(box[2]/2))]
                    output_path = os.path.join(output_video_folder, f"{frame[:-4]}_{str(count).zfill(2)}.jpg")
                    count += 1
                    cv2.imwrite(output_path, crop_img)
                except:
                    continue

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script to export every crop in a given set.")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=False,
        default="./yolov8x.pt",
        help="Path to the model used for detection.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="../../data/SoccerNetGS/challenge",
        help="Path to the data (train, valid, test or challenge).",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        default="./bbox_detections",
        help="Path to export the crops.",
    )
    parser.add_argument(
        "--videos_list",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="List of videos to use.",
    )
    args = vars(parser.parse_args())
    model_path = args["model_path"]
    data_path = args["data_path"]
    output_folder = args["output_folder"]
    videos_list = args["videos_list"]

    extract_persons_crops_from_videos(model_path, data_path, output_folder, videos_list)
