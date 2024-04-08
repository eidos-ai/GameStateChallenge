import argparse
from pathlib import Path
from ultralytics import YOLO  # type: ignore
from gsr.utils.neptune_logger import NeptuneLogger


class ValidateYOLO:
    def __init__(
        self,
        base_model: str,
        experiments: str,
        name: str,
        eval_conf_th: float,
        yaml: str,
        project: str,
        neptune: bool,
        batch_size: int,
        neptune_logger: NeptuneLogger = None,  # type: ignore
    ) -> None:
        self.experiments = experiments
        self.base_model = base_model
        self.name = name
        self.eval_conf_th = eval_conf_th
        self.yaml = yaml
        self.project = project
        self.neptune = neptune
        self.batch_size = batch_size
        self.neptune_logger = neptune_logger

    def validate(self) -> None:
        experiments_dir = Path(self.experiments)

        # Load base model
        model = YOLO(self.base_model)

        # Set experiment name
        model_name = self.base_model.split("/")[-1:][0]
        name = self.name if self.name else f"{model_name}_{self.eval_conf_th}"

        # Set dataset path
        dataset_path = "/".join(self.yaml.split("/")[:-1])

        if self.neptune_logger is None:
            self.neptune_logger = NeptuneLogger(
                enable_run=self.neptune, project=self.project, run_name_or_id=name
            )
        self.neptune_logger.param("dataset path", dataset_path)
        self.neptune_logger.param("conf", self.eval_conf_th)
        self.neptune_logger.param("batch size", self.batch_size)
        self.neptune_logger.param("Yolo model", self.base_model)
        self.experiment_id = self.neptune_logger.get_id()
        self.neptune_logger.tags("ValYOLO")

        results_path = Path(experiments_dir, self.experiment_id)

        # Validate the model
        val_metrics = model.val(
            data=self.yaml,
            batch=self.batch_size,
            conf=self.eval_conf_th,
            project=str(results_path),
        )

        results_path = results_path.joinpath("val")

        images_names = [
            "P_curve.png",
            "PR_curve.png",
            "R_curve.png",
        ]

        for im_name in images_names:
            im_path = results_path.joinpath(im_name)
            self.neptune_logger.upload_file(im_name, im_path)

        # Precision and Recall
        conf_matrix = val_metrics.confusion_matrix.matrix
        for class_idx, class_name in val_metrics.names.items():
            if class_name == "person":
                class_P = (
                    conf_matrix[class_idx, class_idx] / conf_matrix[class_idx, :].sum()
                )
                class_R = (
                    conf_matrix[class_idx, class_idx] / conf_matrix[:, class_idx].sum()
                )
                self.neptune_logger.log(f"{class_name} Precision", round(class_P, 2))
                self.neptune_logger.log(f"{class_name} Recall", round(class_R, 2))

        # Other metrics
        for metric_name, metric_value in val_metrics.results_dict.items():
            self.neptune_logger.log(metric_name, metric_value)

        self.neptune_logger.stop()


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
    validate_yolo = ValidateYOLO(
        args.base_model,
        args.experiments,
        args.name,
        args.eval_conf_th,
        args.yaml,
        args.project,
        args.neptune,
        args.batch_size,
    )
    validate_yolo.validate()
