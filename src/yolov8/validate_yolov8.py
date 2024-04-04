import argparse
from pathlib import Path
from ultralytics import YOLO  # type: ignore
from gsr.utils.neptune_logger import NeptuneLogger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--base_model",
        required=True,
        default="GameStateChallenge/pretrained_models/yolo/yolov8x6.pt",
        help="The base yolo model to use for training. Default value is the baseline model",
    )
    parser.add_argument(
        "-y",
        "--yaml",
        required=True,
        help="Path to the yaml file containing class names and paths to train and val datasets",
    )
    parser.add_argument(
        "--name", required=False, type=str, help="The name of the experiment run"
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        type=int,
        default=16,
        help="training batch size",
    )
    parser.add_argument(
        "-c",
        "--eval_conf_th",
        required=False,
        type=float,
        default=0.5,
        help="Confidence score used for evaluation of model",
    )
    parser.add_argument(
        "--neptune",
        required=False,
        action="store_true",
        default=False,
        help="Log experiment results to neptune",
    )
    parser.add_argument(
        "--experiments",
        required=False,
        type=str,
        default="experiments/yolov8",
        help="Path to the directory where experiments are stored",
    )
    parser.add_argument(
        "--project",
        required=False,
        type=str,
        default="eidos/GameStateChallenge",
        help="The name of the project",
    )

    args = parser.parse_args()
    experiments_dir = Path(args.experiments)

    # Load base model
    model = YOLO(args.base_model)

    # Set experiment name
    name = args.name if args.name else f"{args.base_model}_{args.eval_conf_th}"

    # Set dataset path
    yaml_path = args.yaml
    dataset_path = "/".join(yaml_path.split("/")[:-1])

    # Add training parameters to neptune
    neptune_logger = NeptuneLogger(
        enable_run=args.neptune, project=args.project, run_name_or_id=name
    )
    neptune_logger.param("dataset path", dataset_path)
    neptune_logger.param("conf", args.eval_conf_th)
    neptune_logger.param("batch size", args.batch_size)
    neptune_logger.param("Yolo model", args.base_model)
    experiment_id = neptune_logger.get_id()
    neptune_logger.tags("Eval GameState")

    # Validate the model
    validation_results = model.val(
        data=args.yaml,
        batch=args.batch_size,
        conf=args.eval_conf_th,
    )
    results_path = Path(experiments_dir, experiment_id)

    print("-------------------")
    print(validation_results)
    print("-------------------")

    neptune_logger.stop()
