import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cluster_classification_dataset import ClusterClassificationDataset
from cluster_feats import ClusterFeats
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score


def run_model(model: str, train_dataset: ClusterClassificationDataset, valid_dataset: ClusterClassificationDataset,
              cluster_feats: ClusterFeats, rf_n_estimators: int=100, rf_max_depth: int=None, rf_criterion: str="gini",
              xgb_n_estimators: int=6, xgb_max_depth: int=2, xgb_learning_rate: float=1,
              xgb_objective: str="binary:logistic") -> None:
    X_train = []
    y_train = []
    for data in tqdm(train_dataset, total=len(train_dataset), desc='Iterating over train dataset'):
        feats = cluster_feats.forward(data)
        arr = feats.to_vector()
        X_train.append(arr)
        label = None
        if data.is_valid:
            label = data.this_tracks_role_id
        else:
            label = 6
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_valid = []
    y_valid = []
    for data in tqdm(valid_dataset, total=len(valid_dataset), desc='Iterating over valid dataset'):
        feats = cluster_feats.forward(data)
        arr = feats.to_vector()
        X_valid.append(arr)
        label = None
        if data.is_valid:
            label = data.this_tracks_role_id
        else:
            label = 6
        y_valid.append(label)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)
    
    title = None
    if model == "random_forest":
        model_instance = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                                    criterion=rf_criterion)  
        title = "Random Forest"
    elif model == "xgboost":
        model_instance = XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth,
                            learning_rate=xgb_learning_rate, objective=xgb_objective)
        title = "XGBoost"
    else:
        raise Exception

    model_instance.fit(X_train, y_train)
    y_pred = model_instance.predict(X_valid)

    accuracy = round(accuracy_score(y_valid, y_pred), 4)
    precision = round(precision_score(y_valid, y_pred, average='weighted'), 4)
    recall = round(recall_score(y_valid, y_pred, average='weighted'), 4)

    cm = confusion_matrix(y_valid, y_pred)
    fig, axs = plt.subplots(figsize=(10, 10))
    display_labels = ["pl_lt", "pl_rt", "gk_lt",
                      "gke_rt", "ref", "oth", "inv"]
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(ax=axs)
    axs.set_title(f'Model {title} (acc: {accuracy})')
    plt.show()

    print(f"{model} accuracy:", accuracy)
    print(f"{model} precision:", precision)
    print(f"{model} recall:", recall)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classification model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Options: 'random_forest' and 'xgboost'.",
    )
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
        "--homogeneity_threshold",
        type=float,
        required=False,
        default=0.9,
        help="Homogeneity threshold for valid ClusterScenario instances.",
    )
    parser.add_argument(
        "--scenarios_per_video",
        type=int,
        required=False,
        default=100,
        help="The amount of valid plus invalid ClusterScenario instances generated per video.",
    )
    parser.add_argument(
        "--rf_n_estimators",
        type=int,
        required=False,
        default=100,
        help="Random Forest: The number of trees in the forest.",
    )
    parser.add_argument(
        "--rf_max_depth",
        type=int,
        required=False,
        default=None,
        help="Random Forest: The maximum depth of the tree.",
    )
    parser.add_argument(
        "--rf_criterion",
        type=str,
        required=False,
        default="gini",
        help="Random Forest: The function to measure the quality of a split. Supported criteria are: 'gini', 'entropy' and 'log_loss'.",
    )
    parser.add_argument(
        "--xgb_n_estimators",
        type=int,
        required=False,
        default=6,
        help="XGBoost: Number of trees (or rounds) in the model.",
    )
    parser.add_argument(
        "--xgb_max_depth",
        type=int,
        required=False,
        default=2,
        help="XGBoost: Maximum depth of the trees.",
    )
    parser.add_argument(
        "--xgb_learning_rate",
        type=float,
        required=False,
        default=1,
        help="XGBoost: Learning rate of the model.",
    )
    parser.add_argument(
        "--xgb_objective",
        type=str,
        required=False,
        default="binary:logistic",
        help="XGBoost: Learning task and corresponding learning objective.",
    )

    args = vars(parser.parse_args())

    # choose model type
    model = args["model"]

    # datasets params
    train_data_path = args["train_data_path"]
    train_videos_list = args["train_videos_list"]
    valid_data_path = args["valid_data_path"]
    valid_videos_list = args["valid_videos_list"]
    homogeneity_threshold = args["homogeneity_threshold"]
    scenarios_per_video = args["scenarios_per_video"]

    # random forest classifier hyper parameters
    rf_n_estimators = args["rf_n_estimators"]
    rf_max_depth = args["rf_max_depth"]
    rf_criterion = args["rf_criterion"]

    # xgboost hyper parameters xgb_
    xgb_n_estimators = args["xgb_n_estimators"]
    xgb_max_depth = args["xgb_max_depth"]
    xgb_learning_rate = args["xgb_learning_rate"]
    xgb_objective = args["xgb_objective"]

    # instanciate datasets
    train_dataset = ClusterClassificationDataset(train_data_path, train_videos_list, homogeneity_threshold, scenarios_per_video)
    valid_dataset = ClusterClassificationDataset(valid_data_path, valid_videos_list, homogeneity_threshold, scenarios_per_video)

    # instanciate feats extractor
    cluster_feats = ClusterFeats()

    # execute script's main function
    run_model(model, train_dataset, valid_dataset, cluster_feats, rf_n_estimators, rf_max_depth, rf_criterion, xgb_n_estimators, xgb_max_depth, xgb_learning_rate, xgb_objective)
