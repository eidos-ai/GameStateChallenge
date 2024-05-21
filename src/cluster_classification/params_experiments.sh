#!/bin/bash
scenarios=("100" "150" "200" "250" "300" "350")

for i in ${!scenarios[@]}; do
  python cluster_classification_model.py --train_datasets synth_train synth_valid real_valid --valid_datasets real_test real_valid --valid_real_cluster_path GAM226clusters_valid.json --test_real_cluster_path GAM226clusters_test.json --model random_forest --neptune --rf_n_estimators 200 --rf_criterion entropy --scenarios_per_video ${scenarios[$i]}
done