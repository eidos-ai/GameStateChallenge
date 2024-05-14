import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


def run_random_forest(X_train: np.array, y_train: np.array, X_valid: np.array, y_valid: np.array) -> None:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_valid)

    cm = confusion_matrix(y_valid, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot();

    accuracy = round(accuracy_score(y_valid, y_pred), 4)
    precision = round(precision_score(y_valid, y_pred, average='weighted'), 4)
    recall = round(recall_score(y_valid, y_pred, average='weighted'), 4)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from cluster_classification_dataset import ClusterClassificationDataset
    from cluster_feats import ClusterFeats

    parser = argparse.ArgumentParser(description="Random forest classification model.")
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=False,
        default="../../data/SoccerNetGS/train",
        help="Path to the train set.",
    )
    parser.add_argument(
        "--train_videos_list",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="List of videos to train.",
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        required=False,
        default="../../data/SoccerNetGS/valid",
        help="Path to the valid set.",
    )
    parser.add_argument(
        "--valid_videos_list",
        type=str,
        required=False,
        nargs="+",
        default=None,
        help="List of videos for validation.",
    )

    args = vars(parser.parse_args())
    train_data_path = args["train_data_path"]
    train_videos_list = args["train_videos_list"]
    valid_data_path = args["valid_data_path"]
    valid_videos_list = args["valid_videos_list"]

    train_dataset = ClusterClassificationDataset(train_data_path, train_videos_list)
    valid_dataset = ClusterClassificationDataset(valid_data_path, valid_videos_list)
    cluster_feats = ClusterFeats()

    X_train = []
    y_train = []
    for data, label in tqdm(train_dataset, total=len(train_dataset), desc='Iterating over train dataset'):
        feats = cluster_feats.forward(data)
        arr = feats.to_vector()
        X_train.append(arr)
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_valid = []
    y_valid = []
    for data, label in tqdm(valid_dataset, total=len(valid_dataset), desc='Iterating over valid dataset'):
        feats = cluster_feats.forward(data)
        arr = feats.to_vector()
        X_valid.append(arr)
        y_valid.append(label)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)

    run_random_forest(X_train, y_train, X_valid, y_valid)
