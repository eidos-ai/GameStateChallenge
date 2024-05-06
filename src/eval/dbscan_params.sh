VIDEO="SNGS-021 SNGS-039 SNGS-078"
VIDEO_SET=valid

OUTPUT_PATH_GAM199=../output/tmp/GAM-199_embeddings
OUTPUT_PATH_GAM200=../output/tmp/GAM-200_embeddings

EPS=0.6 # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
MIN_SAMPLES=3 # The number of samples in a neighborhood for a point to be considered as a core point.

cd ..
echo "Test with eps=${EPS} and min_samples=${MIN_SAMPLES}:"
echo "Evaluation on GAM199-epoch2.pt:"
python eval/dbscan_evaluation.py --data_path ../data/SoccerNetGS/$VIDEO_SET --embeddings_directory $OUTPUT_PATH_GAM199 --videos_list $VIDEO --eps $EPS --min_samples $MIN_SAMPLES
echo "Evaluation on GAM200-epoch_1.pt:"
python eval/dbscan_evaluation.py --data_path ../data/SoccerNetGS/$VIDEO_SET --embeddings_directory $OUTPUT_PATH_GAM200 --videos_list $VIDEO --eps $EPS --min_samples $MIN_SAMPLES