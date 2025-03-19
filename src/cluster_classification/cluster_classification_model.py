import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Optional
import joblib
from torch.utils.data import ConcatDataset
from dataclasses import dataclass
from cluster_classification_dataset import ClusterClassificationDataset
from real_clusters_dataset import RealClustersDataset
from SNGS_dataset.groundtruth_reader import GroundTruthReader 
from evaluate_model_acc import evaluate_clustering

from cluster_feats import ClusterFeats
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score

from utils.neptune_logger import NeptuneLogger

@dataclass
class DatasetConfiguration:
    # Paths to take GroundTruth Datasets. These are used to generate SynthDatasets
    train_data_path: str
    valid_data_path: str
    test_data_path: str
    
    # This limits the dataset videos
    train_videos_list: List[str]
    valid_videos_list: List[str]
    
    # These datasets are real clusterings created by running the embeddings model and different KMeans
    valid_clustering_path: Optional[str]
    test_clustering_path: Optional[str]
    
    # These are parameters for generating synth data of clusterings
    homogeneity_threshold: float
    scenarios_per_video: int
    
    # These defines which datasets are used to train and which to eval the model
    # Options are: [synth_train, synth_valid, synth_test, real_valid, real_test]
    train_datasets: List[str]
    valid_datasets: List[str]
    
class DatasetRegistry:
    def __init__(self, conf: DatasetConfiguration):
        self.conf = conf
        
        print("Reading train")
        train_dataset = ClusterClassificationDataset(conf.train_data_path, conf.train_videos_list, conf.homogeneity_threshold, conf.scenarios_per_video)

        print("Reading valid")
        valid_dataset = ClusterClassificationDataset(conf.valid_data_path, conf.valid_videos_list, conf.homogeneity_threshold, conf.scenarios_per_video)

        print("Reading test")
        test_dataset = ClusterClassificationDataset(conf.test_data_path, None, conf.homogeneity_threshold, conf.scenarios_per_video)

        if conf.valid_clustering_path:
            valid_clustering_dataset = RealClustersDataset(conf.valid_data_path, conf.valid_clustering_path)
        else:
            valid_clustering_dataset = None
        if conf.test_clustering_path:
            test_clustering_dataset = RealClustersDataset(conf.test_data_path, conf.test_clustering_path)
        else:
            test_clustering_dataset = None                    
        
        self.registry = {
            'synth_train': train_dataset,
            'synth_valid': valid_dataset,
            'synth_test': test_dataset,
            'real_valid': valid_clustering_dataset,
            'real_test': test_clustering_dataset
        }
    
    
    def get_train_dataset(self):
        train_datasets = ConcatDataset([self.registry[dataset] for dataset in self.conf.train_datasets])
        return train_datasets
    
    def get_valid_dataset(self):
        valid_datasets = [(dataset, self.registry[dataset]) for dataset in self.conf.valid_datasets]
        return valid_datasets        
      
def evaluate_model(model_instance, dataset_name: str, dataset: Dataset, cluster_feats, neptune_logger: NeptuneLogger):
    X = []
    y = []
    for data in tqdm(dataset, total=len(dataset), desc=f'Iterating over {dataset_name} dataset'):
        feats = cluster_feats.forward(data.this_tracks, data.rest_tracks)
        arr = feats.to_vector()
        X.append(arr)
        label = None
        if data.is_valid:
            label = data.this_tracks_role_id
        else:
            label = 6
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    
    y_pred = model_instance.predict(X)

    accuracy = round(accuracy_score(y, y_pred), 4)
    precision = round(precision_score(y, y_pred, average='weighted'), 4)
    recall = round(recall_score(y, y_pred, average='weighted'), 4)

    # cm = confusion_matrix(y, y_pred)
    # fig, axs = plt.subplots(figsize=(10, 10))
    # display_labels = ["pl_lt", "pl_rt", "gk_lt",
    #                   "gke_rt", "ref", "oth", "inv"]
    # ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(ax=axs)
    # axs.set_title(f'Model on Dataset {dataset_name} (acc: {accuracy})')
    # plt.show()

    # print(f"{dataset_name} - accuracy:", accuracy)
    # print(f"{dataset_name} - precision:", precision)
    # print(f"{dataset_name} - recall:", recall)
    neptune_logger.log(f"evaluation/{dataset_name}/accuracy", accuracy)
    neptune_logger.log(f"evaluation/{dataset_name}/precision", precision)
    neptune_logger.log(f"evaluation/{dataset_name}/recall", recall)


def run_model(model: str, neptune_logger: NeptuneLogger, store_model: str, train_dataset: Dataset, valid_dataset: List[Tuple[str, Dataset]], 
                                                                                        
              cluster_feats: ClusterFeats, rf_n_estimators: int=100, rf_max_depth: int=None, rf_criterion: str="gini",
              xgb_n_estimators: int=6, xgb_max_depth: int=2, xgb_learning_rate: float=1,
              xgb_objective: str="binary:logistic") -> None:
    # Train dataset
    X_train = []
    y_train = []
    for data in tqdm(train_dataset, total=len(train_dataset), desc='Iterating over train dataset'):
        feats = cluster_feats.forward(data.this_tracks, data.rest_tracks)
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
    
    
    # Define model
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
    
    # Fit
    model_instance.fit(X_train, y_train)
    
    # Evaluate
    for dataset_name, dataset in valid_dataset:
        evaluate_model(model_instance, dataset_name, dataset,  cluster_feats, neptune_logger)
        if isinstance(dataset, RealClustersDataset):
            track_metrics = evaluate_clustering(dataset, model_instance)
            neptune_logger.log(f"evaluation/{dataset_name}/percent_correct_clustering",
                               track_metrics.percent_correct_clustering)
            neptune_logger.log(f"evaluation/{dataset_name}/average_track_accuracy",
                               track_metrics.average_track_accuracy * 100)
            neptune_logger.log(f"evaluation/{dataset_name}/average_bbox_accuracy",
                               track_metrics.average_bbox_accuracy * 100)
            neptune_logger.log(f"evaluation/{dataset_name}/average_best_track_accuracy",
                               track_metrics.average_best_track_accuracy * 100)
            neptune_logger.log(f"evaluation/{dataset_name}/average_best_bbox_accuracy",
                               track_metrics.average_best_bbox_accuracy * 100)
            print("///////////////////////////////////////////////////////////////////////////")
            print(f'{dataset_name} metrics:')
            print("\n")
            # for metrics in track_metrics.video_metrics:
                # print(metrics)
                # print("\n")
            print("///////////////////////////////////////////////////////////////////////////")
            print(f"///// Evaluation Clustering on {dataset_name}")
            print(f"% of videos with correct clustering found: {track_metrics.percent_correct_clustering:.2f}%")
            print(f"Average Track role_id Accuracy {track_metrics.average_track_accuracy * 100:.2f}")
            print(f"Average Bbox role_id Accuracy {track_metrics.average_bbox_accuracy * 100:.2f}")
            print(f"True Clustering Track role_id Accuracy {track_metrics.average_best_track_accuracy * 100:.2f}")
            print(f"True Clustering Average Bbox role_id Accuracy {track_metrics.average_best_bbox_accuracy * 100:.2f}")

    # Save model
    if store_model is not None:
        joblib.dump(model_instance, store_model)
        print(f"Saved model to {store_model}")


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
        "--test_data_path",
        type=str,
        required=False,
        default="../../data/SoccerNetGS/test",
        help="Path to the test set.",
    )
    parser.add_argument(
        "--valid_real_cluster_path",
        type=str,
        required=False,
        default=None,
        help="Path to a json with RealClustering for Valid Dataset",
    )
    parser.add_argument(
        "--test_real_cluster_path",
        type=str,
        required=False,
        default=None,
        help="Path to a json with RealClustering for Test Dataset",
    )                                                                                    
    parser.add_argument(
        "--train_datasets",
        nargs="+",
        type=str,
        help="List of training datasets"
    )
    parser.add_argument(
        "--valid_datasets",
        nargs="+",
        type=str,
        help="List of valid datasets"
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
    parser.add_argument(
        "--store_model",
        type=str,
        required=False,
        default=None,
        help="Path to store trained model",
    )
    parser.add_argument(
        "--neptune",
        required=False,
        action="store_true",
        default=False,
        help="Log experiment results to neptune"
    )
    parser.add_argument(
        "--project",
        required=False,
        type=str,
        default="eidos/GameStateChallenge",
        help="The name of the project"
    )
    parser.add_argument(
        "--name",
        required=False,
        type=str,
        help="The name of the experiment"
    )

    args = vars(parser.parse_args())

    # choose model type
    model = args["model"]

    # datasets params
    train_data_path = args["train_data_path"]
    train_videos_list = args["train_videos_list"]
    valid_data_path = args["valid_data_path"]
    valid_videos_list = args["valid_videos_list"]
    test_data_path = args["test_data_path"]
    valid_real_cluster_path = args["valid_real_cluster_path"]
    test_real_cluster_path = args["test_real_cluster_path"]
    train_datasets = args["train_datasets"]
    valid_datasets = args["valid_datasets"]
    homogeneity_threshold = args["homogeneity_threshold"]
    scenarios_per_video = args["scenarios_per_video"]

    # random forest classifier hyper parameters
    rf_n_estimators = args["rf_n_estimators"]
    rf_max_depth = args["rf_max_depth"]
    rf_criterion = args["rf_criterion"]

    # xgboost hyper parameters xgboost
    xgb_n_estimators = args["xgb_n_estimators"]
    xgb_max_depth = args["xgb_max_depth"]
    xgb_learning_rate = args["xgb_learning_rate"]
    xgb_objective = args["xgb_objective"]

    store_model = args["store_model"]
    
    # instanciate datasets
    data_conf = DatasetConfiguration(
        train_data_path, valid_data_path,test_data_path,
        train_videos_list, valid_videos_list,
        valid_real_cluster_path, test_real_cluster_path,
        homogeneity_threshold, scenarios_per_video,
        train_datasets, valid_datasets
    )
    
    # add parameters to neptune
    neptune_logger = NeptuneLogger(enable_run=args["neptune"], project=args["project"], run_name_or_id=args["name"])
    neptune_logger.tags(["ClusterClassification"])

    neptune_logger.param("model/method", model)
    neptune_logger.param("valid_real_cluster_path", valid_real_cluster_path)
    neptune_logger.param("test_real_cluster_path", test_real_cluster_path)
    neptune_logger.param("train_datasets/datasets", str(train_datasets))
    neptune_logger.param("valid_datasets/datasets", str(valid_datasets))

    if "synth_train" in train_datasets:
        neptune_logger.param("train_datasets/synth_train/homogeneity_threshold", homogeneity_threshold)
        neptune_logger.param("train_datasets/synth_train/scenarios_per_video", scenarios_per_video)
    if "synth_train" in valid_datasets:
        neptune_logger.param("valid_datasets/synth_train/homogeneity_threshold", homogeneity_threshold)
        neptune_logger.param("valid_datasets/synth_train/scenarios_per_video", scenarios_per_video)
        
    if "synth_valid" in train_datasets:
        neptune_logger.param("train_datasets/synth_valid/homogeneity_threshold", homogeneity_threshold)
        neptune_logger.param("train_datasets/synth_valid/scenarios_per_video", scenarios_per_video)
    if "synth_valid" in valid_datasets:
        neptune_logger.param("valid_datasets/synth_valid/homogeneity_threshold", homogeneity_threshold)
        neptune_logger.param("valid_datasets/synth_valid/scenarios_per_video", scenarios_per_video)
        
    if "synth_test" in train_datasets:
        neptune_logger.param("train_datasets/synth_test/homogeneity_threshold", homogeneity_threshold)
        neptune_logger.param("train_datasets/synth_test/scenarios_per_video", scenarios_per_video)
    if "synth_test" in valid_datasets:
        neptune_logger.param("valid_datasets/synth_test/homogeneity_threshold", homogeneity_threshold)
        neptune_logger.param("valid_datasets/synth_test/scenarios_per_video", scenarios_per_video)
        
    if model == "random_forest":
        neptune_logger.param("model/n_estimators", rf_n_estimators)
        if rf_max_depth is None:
            neptune_logger.param("model/max_depth", "None (default)")
        else:
            neptune_logger.param("model/max_depth", rf_max_depth)
        neptune_logger.param("model/criterion", rf_criterion)
    elif model == "xgboost":
        neptune_logger.param("model/n_estimators", xgb_n_estimators)
        neptune_logger.param("model/max_depth", xgb_max_depth)
        neptune_logger.param("model/learning_rate", xgb_learning_rate)
        neptune_logger.param("model/objective", xgb_objective)

    data_registry = DatasetRegistry(data_conf)
    train_datasets = data_registry.get_train_dataset()
    valid_dataset = data_registry.get_valid_dataset()
    
    # instanciate feats extractor
    cluster_feats = ClusterFeats()

    # execute script's main function
    run_model(model, neptune_logger, store_model, train_datasets, valid_dataset, cluster_feats, rf_n_estimators, rf_max_depth, rf_criterion, xgb_n_estimators, xgb_max_depth, xgb_learning_rate, xgb_objective)
