#!/bin/bash
models=("random_forest" "xgboost")
model_files=("model_rf.pkl" "model_xgb.pkl")

for i in ${!models[@]}; do
  python cluster_classification_model.py --train_datasets synth_train synth_valid real_valid --valid_datasets real_test real_valid --valid_real_cluster_path GAM226clusters_valid.json --test_real_cluster_path GAM226clusters_test.json --model ${models[$i]} --store_model ${model_files[$i]} --neptune
done