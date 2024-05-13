import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


def run_random_forest(X: np.array, y: np.array) -> None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 4)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from cluster_classification_dataset import ClusterClassificationDataset
    from cluster_feats import ClusterFeats

    parser = argparse.ArgumentParser(description="Random forest classification model.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="../../data/SoccerNetGS/challenge",
        help="Path to the data (train, valid, test or challenge).",
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
    data_path = args["data_path"]
    videos_list = args["videos_list"]

    dataset = ClusterClassificationDataset(data_path, videos_list)
    cluster_feats = ClusterFeats()

    X = []
    y = []
    for data, label in tqdm(dataset, total=len(dataset), desc='Iterating over dataset'):
        feats = cluster_feats.forward(data)
        arr = feats.to_vector()
        X.append(arr)
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    run_random_forest(X,y)
