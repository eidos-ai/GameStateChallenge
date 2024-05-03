VIDEO="SNGS-021 SNGS-039 SNGS-078"
VIDEO_SET=valid
MODEL_PATH=../experiments/identifier/GAM-200/GAM200-epoch_1.pt
OUTPUT_PATH=../output/tmp/GAM-200_embeddings
MLP_OUTPUT_SIZE=128
EPS=1
MIN_SAMPLES=2
cd ..
# Run model to get embeddings
python eval/model_run.py -m $MODEL_PATH --videos_list $VIDEO --output_path $OUTPUT_PATH --data_path ../data/SoccerNetGS/$VIDEO_SET --mlp_output_size $MLP_OUTPUT_SIZE

# DBScan
python eval/dbscan_clustering.py --embeddings_directory $OUTPUT_PATH --output_directory $OUTPUT_PATH/dbscan --data_path ../data/SoccerNetGS/$VIDEO_SET --videos_list $VIDEO --eps $EPS --min_samples $MIN_SAMPLES
python eval/dbscan_evaluation.py --data_path ../data/SoccerNetGS/$VIDEO_SET --embeddings_directory $OUTPUT_PATH --videos_list $VIDEO --eps $EPS --min_samples $MIN_SAMPLES

# Magic KMeans
python eval/magic_kmeans_clustering.py --data_path ../data/SoccerNetGS/$VIDEO_SET --embeddings_directory $OUTPUT_PATH --videos_list $VIDEO —output_directory ./output/test1/out_kmeans
python eval/clustering_accuracy_evaluation.py --data_path ../data/SoccerNetGS/$VIDEO_SET --embeddings_directory $OUTPUT_PATH --videos_list $VIDEO