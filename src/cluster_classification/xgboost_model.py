import numpy as np
from tqdm import tqdm
from cluster_classification_dataset import ClusterClassificationDataset
from cluster_feats import ClusterFeats
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score


def run_xgboost_model(train_dataset: ClusterClassificationDataset, valid_dataset: ClusterClassificationDataset,
                      cluster_feats: ClusterFeats, n_estimators: int=6, max_depth: int=2, learning_rate: float=1,
                      objective: str='binary:logistic') -> None:
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

    bst = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, objective=objective)
    bst.fit(X_train, y_train)
    y_pred = bst.predict(X_valid)

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

    parser = argparse.ArgumentParser(description="XGBoost classification model.")
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
    parser.add_argument(
        "--n_estimators",
        type=int,
        required=False,
        default=6,
        help="Number of trees (or rounds) in a XGBoost model.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        required=False,
        default=2,
        help="Maximum depth of the trees.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        default=1,
        help="Learning rate of the model.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        required=False,
        default="binary:logistic",
        help="Learning task and corresponding learning objective.",
    )

    args = vars(parser.parse_args())
    train_data_path = args["train_data_path"]
    train_videos_list = args["train_videos_list"]
    valid_data_path = args["valid_data_path"]
    valid_videos_list = args["valid_videos_list"]
    n_estimators = args["n_estimators"]
    max_depth = args["max_depth"]
    learning_rate = args["learning_rate"]
    objective = args["objective"]

    train_dataset = ClusterClassificationDataset(train_data_path, train_videos_list)
    valid_dataset = ClusterClassificationDataset(valid_data_path, valid_videos_list)
    cluster_feats = ClusterFeats()

    run_xgboost_model(train_dataset, valid_dataset, cluster_feats, n_estimators, max_depth, learning_rate, objective)
