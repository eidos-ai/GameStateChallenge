#!/bin/bash
models=("random_forest" "random_forest" "random_forest" "random_forest" "xgboost" "xgboost" "xgboost" "xgboost")
model_files=("model_rf1.pkl" "model_rf2.pkl" "model_rf3.pkl" "model_rf4.pkl" "model_xgb1.pkl" "model_xgb2.pkl" "model_xgb3.pkl" "model_xgb4.pkl")
n_estimators=("50" "100" "150" "200" "4" "6" "8" "10")
criterion=("gini" "gini" "entropy" "entropy" "." "." "." ".")
objective=("." "." "." "." "reg:squarederror" "reg:absoluteerror" "binary:logistic" "binary:logistic")

for i in ${!models[@]}; do
  python cluster_classification_model.py --train_datasets synth_train synth_valid real_valid --valid_datasets real_test real_valid --valid_real_cluster_path GAM226clusters_valid.json --test_real_cluster_path GAM226clusters_test.json --model ${models[$i]} --store_model ${model_files[$i]} --neptune --rf_n_estimators ${n_estimators[$i]} --rf_criterion ${criterion[$i]} --xgb_n_estimators ${n_estimators[$i]} --xgb_objective ${objective[$i]}
done