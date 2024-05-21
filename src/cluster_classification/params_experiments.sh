#!/bin/bash
models=("random_forest" "random_forest" "random_forest" "random_forest" "random_forest" "random_forest")
n_estimators=("200" "200" "200" "200" "200" "200")
criterion=("entropy" "entropy" "entropy" "entropy" "entropy" "entropy")
scenarios=("100" "150" "200" "250" "300" "350")

for i in ${!models[@]}; do
  python cluster_classification_model.py --train_datasets synth_train synth_valid real_valid --valid_datasets real_test real_valid --valid_real_cluster_path GAM226clusters_valid.json --test_real_cluster_path GAM226clusters_test.json --model ${models[$i]} --neptune --rf_n_estimators ${n_estimators[$i]} --rf_criterion ${criterion[$i]} --scenarios_per_video ${scenarios[$i]}
done