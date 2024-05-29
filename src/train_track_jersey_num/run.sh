# python save_detections.py --model_path ../../models/GAM-412_epoch0.pt  --out_path ../../experiments/track_jersey_num/exp1_valid --data_path ../../data/SoccerNetGS/valid
# python save_detections.py --model_path ../../models/GAM-412_epoch0.pt  --out_path ../../experiments/track_jersey_num/exp1_test --data_path ../../data/SoccerNetGS/test

# python eval.py  --data_path ../../data/SoccerNetGS/valid --saved_detections ../../experiments/track_jersey_num/exp1_valid --neptune
# python eval.py  --data_path ../../data/SoccerNetGS/test --saved_detections ../../experiments/track_jersey_num/exp1_test --neptune

python train_solver.py --train_datasets soccernet_valid --valid_datasets soccernet_test --saved_valid ../../experiments/track_jersey_num/exp1_valid --saved_test ../../experiments/track_jersey_num/exp1_test
python train_solver.py --train_datasets soccernet_test --valid_datasets soccernet_valid --saved_valid ../../experiments/track_jersey_num/exp1_valid --saved_test ../../experiments/track_jersey_num/exp1_test