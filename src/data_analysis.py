import os
from typing import Any

directories_path = "./data/SoccerNetGS/gamestate-2024/"
data_directories = ["challenge", "test", "train", "valid"]


# Print the number of folders and files in every subdirectory of the data directory
def count_folders_and_files(directory: str) -> tuple[int, int, list[str]]:
    folders = 0
    files = 0
    filenames = []

    for name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, name)):
            files += 1
            filenames.append(name)
        else:
            folders += 1

    return (folders, files, filenames)


# Check if sequences_info.json files in different directories are the same
import json

sequences_files_data = []


def check_duplicated_sequence_files() -> Any:
    file_name = "/sequences_info.json"

    for directory in data_directories:
        file_path = os.path.join(directories_path, directory + file_name)
        with open(file_path, "r") as file:
            data = json.load(file)
            sequences_files_data.append(data)

    return (
        sequences_files_data[0]
        == sequences_files_data[1]
        == sequences_files_data[2]
        == sequences_files_data[3]
    )


# In Labels-GameState.json, print every annotation of supercategory "camera" where the image of that annotation has "has_enabled_camera" as False
def print_camera_annotations() -> None:
    labels_file_path = os.path.join(
        directories_path, "test/SNGS-116/Labels-GameState.json"
    )
    with open(labels_file_path, "r") as file:
        data = json.load(file)
        for image in data["images"]:
            if image["has_labeled_camera"] == False:
                for annotation in data["annotations"]:
                    if (
                        annotation["image_id"] == image["image_id"]
                        and annotation["supercategory"] == "camera"
                    ):
                        print(json.dumps(annotation, indent=4))


def check_goalkeeper_jersey() -> None:
    labels_file_path = os.path.join(
        directories_path, "test/SNGS-116/Labels-GameState.json"
    )
    goalkeeper_with_jersey = False
    with open(labels_file_path, "r") as file:
        data = json.load(file)
        for annotation in data["annotations"]:
            try:
                if (
                    annotation["attributes"]["role"] == "goalkeeper"
                    and annotation["attributes"]["jersey"] != None
                ):
                    goalkeeper_with_jersey = True
                    break
            except:
                pass
    (
        print("Some goalkeepers have jersey number.")
        if goalkeeper_with_jersey
        else print("No goalkeeper has jersey number.")
    )


if __name__ == "__main__":

    # for directory in data_directories:
    #     path = os.path.join(directories_path, directory)
    #     folder_count, file_count, file_name = count_folders_and_files(path)
    #     print(f"Directory: {directory}")
    #     print(f"Number of folders: {folder_count}")
    #     print(f"Number of files: {file_count}")
    #     print(f"Names of those files: {file_name} \n")

    # print("Sequences files are the same.\n") if check_duplicated_sequence_files() else print("Sequences files are different.\n")

    print_camera_annotations()
    check_goalkeeper_jersey()
