# Documentation for the cluster_classification scripts and classes

- [random_forest_model.py](#random_forest_modelpy)
- [xgboost_model.py](#xgboost_modelpy)
- [ClusterClassificationDataset](#ClusterClassificationDataset)


## random_forest_model.py 
This script is used to run a Random Forest Classification Model which its input are extracted features as average position, number of tracks in cluster, etc. The output is measured using the model metrics and the ultimate goal is to use the KMeans number of clusters with the best accuracy.

The script does not have any **required** argument.

#### Random Forest hyperparameters:
- **n_estimators** is the number of trees in the forest.
- **max_depth** is the maximum depth of the trees.
- **criterion** is the function to measure the quality of a split. Supported criteria are: 'gini', 'entropy' and 'log_loss'.

## xgboost_model.py
This script is used to run a XGBoost Classification Model which its input are extracted features as average position, number of tracks in cluster, etc. The output is measured using the model metrics and the ultimate goal is to use the KMeans number of clusters with the best accuracy.

The script does not have any **required** argument.

#### XGBoost hyperparameters:
- **n_estimators** is the number of trees (or rounds) in a XGBoost model.
- **max_depth** is the maximum depth of the trees.
- **learning_rate** of the model.
- **objective** is the learning task and the corresponding learning objective.

## ClusterClassificationDataset
This dataset inherits from Torch Dataset and returns ClusterScenario instances.

The class takes the following arguments:
- `data_path` (str): path to the train, test or valid data directory.
- `videos_list` (list[str], default=None): if None, every video from **data_path** is included.
- `homogeneity_threshold` (float): proportion (between 0 and 1) of Track instances from `ClusterScenario.this_tracks` in which `ClusterScenario.this_tracks_role_id == Track.role_id`, to have `ClusterScenario.is_valid == True`.
- `scenarios_per_video` (int): number of valid plus invalid ClusterScenario instances created from each video (currently balanced).