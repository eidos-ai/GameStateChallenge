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
        help="The base yolo model to use for training.",
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
    neptune_logger.tags("ValYOLO")

    results_path = Path(experiments_dir, experiment_id)

    # Validate the model
    val_metrics = model.val(
        data=args.yaml,
        batch=args.batch_size,
        conf=args.eval_conf_th,
        project=str(results_path)
    )
    
    results_path = results_path.joinpath("val")
    
    images_names = [
        "P_curve.png",
        "PR_curve.png",
        "R_curve.png",
    ]
    
    for im_name in images_names:
        im_path = results_path.joinpath(im_name)
        neptune_logger.upload_file(im_name, im_path)
    
    # Precision and Recall
    conf_matrix = val_metrics.confusion_matrix.matrix
    for class_idx, class_name in val_metrics.names.items():
        if class_name == "person":
            class_P = conf_matrix[class_idx, class_idx] / conf_matrix[class_idx, :].sum()
            class_R = conf_matrix[class_idx, class_idx] / conf_matrix[:, class_idx].sum()
            neptune_logger.log(f"{class_name} Precision", round(class_P, 2))
            neptune_logger.log(f"{class_name} Recall", round(class_R, 2))
            
    # Other metrics
    for metric_name, metric_value in val_metrics.results_dict.items():
        neptune_logger.log(metric_name, metric_value)

    neptune_logger.stop()
