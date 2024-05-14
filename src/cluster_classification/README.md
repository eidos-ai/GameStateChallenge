# Documentation for the cluster_classification scripts

- [random_forest_classification_model.py](#random_forest_classification_modelpy)

## random_forest_classification_model.py 
This script is used to run a Random Forest Classification Model which its input are extracted features as average position, number of tracks in cluster, etc. The output is measured using the model metrics and the ultimate goal is to use the KMeans number of clusters with the best accuracy.

The script takes the following **required** arguments:
- `--data_path` (str): path to the train, test or valid data directory.

An example of a command to run the script is:
```python random_forest_classification_model.py --data_path ../../data/SoccerNetGS/valid --videos_list SNGS-021 SNGS-039 SNGS-078```