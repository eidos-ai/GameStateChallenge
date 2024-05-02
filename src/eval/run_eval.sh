VIDEO=SNGS-060
VIDEO_SET=train
MODEL_PATH=../experiments/identifier/GAM-131/trained_model.pt
OUTPUT_PATH=../output/tmp
cd ..
# Run Model to get Embeddings
python eval/model_run.py -m $MODEL_PATH --videos_list $VIDEO --output_path $OUTPUT_PATH --data_path ../data/SoccerNetGS/$VIDEO_SET
# DB Scan
python eval/dbscan_clustering.py --embeddings_directory $OUTPUT_PATH --output_directory $OUTPUT_PATH/dbscan --data_path ../data/SoccerNetGS/$VIDEO_SET --videos_list $VIDEO --eps 3 --min_samples 3
python eval/dbscan_evaluation.py --data_path ../data/SoccerNetGS/$VIDEO_SET --embeddings_directory $OUTPUT_PATH --videos_list $VIDEO --eps 1 --min_samples 2
# KMeans
python eval/magic_kmeans_clustering.py --data_path ../data/SoccerNetGS/$VIDEO_SET --embeddings_directory $OUTPUT_PATH --videos_list $VIDEO —output_directory ./output/test1/out_kmeans
python eval/clustering_accuracy_evaluation.py --data_path ../data/SoccerNetGS/$VIDEO_SET --embeddings_directory $OUTPUT_PATH --videos_list $VIDEO

