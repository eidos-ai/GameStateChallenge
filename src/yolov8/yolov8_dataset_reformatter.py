import os
import json
import shutil

DIRECTORIES_PATH = "data/SoccerNetGS/gamestate-2024/"


def create_yolov8_dataset_folders(folder: str) -> None:
    """Create the YOLOv8 dataset folders"""
    os.makedirs(f"yolov8_dataset/{folder}/images", exist_ok=True)
    os.makedirs(f"yolov8_dataset/{folder}/labels", exist_ok=True)
    print("Done: the YOLOv8 dataset folders were successfully created.")


def format_images(folder: str) -> None:
    if folder == "val":
        folder = "valid"
    folder_path = os.path.join(DIRECTORIES_PATH, folder)
    for root, dirs, files in os.walk(folder_path):
        for document in files:
            if document.endswith(".jpg"):
                image_path = os.path.join(root, document)
                new_image_name = ""
                index = image_path.find("SNGS-")
                if index != -1:
                    new_image_name = (
                        image_path[index : index + 8] + "-" + image_path[-10:-4]
                    )
                else:
                    exit()
                if folder == "valid":
                    folder = "val"
                new_image_path = (
                    "yolov8_dataset/" + folder + "/images/" + new_image_name + ".jpg"
                )
                shutil.copy2(os.path.join(root, document), new_image_path)
    print(
        f"Done: the images were successfully exported into {folder} folder in YOLOv8 dataset."
    )


def create_annotation_txt_file(folder: str) -> None:
    """from Labels-GameState.json, get the annotations of the corresponding image and
    write them to a txt file in YOLOv8 format"""
    if folder == "val":
        folder = "valid"
    folder_path = os.path.join(DIRECTORIES_PATH, folder)
    for root, dirs, files in os.walk(folder_path):
        for document in files:
            if document.endswith("Labels-GameState.json"):
                with open(os.path.join(root, document)) as json_file:
                    data = json.load(json_file)
                image_ids = set()

                for annotation in data["annotations"]:
                    image_ids.add(annotation["image_id"])
                for image_id in image_ids:
                    formatted_annotations = []
                    for annotation in data["annotations"]:
                        if annotation["image_id"] == image_id and (
                            annotation["category_id"] == 1
                            or annotation["category_id"] == 2
                            or annotation["category_id"] == 3
                        ):
                            formatted_annotation = (
                                "1 "
                                + str(annotation["bbox_image"]["x_center"])
                                + " "
                                + str(annotation["bbox_image"]["y_center"])
                                + " "
                                + str(annotation["bbox_image"]["w"])
                                + " "
                                + str(annotation["bbox_image"]["h"])
                            )
                            formatted_annotations.append(formatted_annotation)
                    json_file_path = os.path.join(root, document)
                    new_txt_name = ""
                    index = json_file_path.find("SNGS-")
                    if index != -1:
                        new_txt_name = (
                            json_file_path[index : index + 8]
                            + "-"
                            + image_id[-6:]
                            + ".txt"
                        )
                    else:
                        exit()
                    if folder == "valid":
                        folder = "val"
                    new_txt_path = (
                        "yolov8_dataset/" + folder + "/labels/" + new_txt_name
                    )
                    with open(new_txt_path, "w") as file:
                        for ann in formatted_annotations:
                            file.write(ann + "\n")
    print(
        f"Done: the labels txt files were successfully exported into {folder} folder in YOLOv8 dataset."
    )


if __name__ == "__main__":
    folders = ["train", "test", "val"]
    for folder in folders:
        create_yolov8_dataset_folders(folder)
        format_images(folder)
        create_annotation_txt_file(folder)
