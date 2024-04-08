import argparse
from pathlib import Path
from ultralytics import YOLO  # type: ignore
from gsr.utils.neptune_logger import NeptuneLogger
from validate_yolov8 import ValidateYOLO


class TrainYOLO:
    def __init__(self, base_model: str, experiments: str, name: str, eval_conf_th: float, yaml: str, project: str, neptune: bool, batch_size: int, epochs: int, initial_learning_rate: float, final_learning_rate: float):
        self.base_model = base_model
        self.experiments = experiments
        self.name = name
        self.eval_conf_th = eval_conf_th
        self.yaml = yaml
        self.project = project
        self.neptune = neptune
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
                

    def train(self):
        experiments_dir = Path(self.experiments)

        # Load base model
        model = YOLO(
            self.base_model
        )  # load a pretrained model (recommended for training)

        name = (
            self.name
            if self.name
            else f"{self.base_model}_{self.epochs}_{self.eval_conf_th}"
        )

        # dataset path
        dataset_path = "/".join(self.yaml.split("/")[:-1])

        # Add training parameters to neptune
        neptune_logger = NeptuneLogger(
            enable_run=self.neptune, project=self.project, run_name_or_id=name
        )
        neptune_logger.param("dataset path", dataset_path)
        neptune_logger.param("epochs", self.epochs)
        neptune_logger.param("Eval conf th", self.eval_conf_th)
        neptune_logger.param("batch size", self.batch_size)
        neptune_logger.param("Yolo model", self.base_model)
        neptune_logger.param("lr0", self.initial_learning_rate)
        neptune_logger.param("lrf", self.final_learning_rate)
        experiment_id = neptune_logger.get_id()
        neptune_logger.tags("TrainYOLO")

        # Start training
        model.train(
            data=self.yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            name=experiment_id,
            project=str(experiments_dir),
            lr0=self.initial_learning_rate,
            lrf=self.final_learning_rate,
        )  # train the model
        results_path = Path(experiments_dir, experiment_id)

        # Upload confusion matrix and other files
        images_names = [
            "confusion_matrix.png",
            "F1_curve.png",
            "labels.jpg",
            "P_curve.png",
            "PR_curve.png",
            "R_curve.png",
            "results.png",
        ]
        for im_name in images_names:
            im_path = results_path.joinpath(im_name)
            neptune_logger.upload_file(im_name, im_path)

        # Perfom evaluation on validation dataset        
        validate_yolo = ValidateYOLO(self.base_model, self.experiments, self.name, self.eval_conf_th, self.yaml, self.project, self.neptune, self.batch_size, neptune_logger)
        validate_yolo.validate()

        # Stop NeptuneLogger instance
        neptune_logger.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--base_model",
        required=True,
        help="The base yolo model to use for training",
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
        "-e",
        "--epochs",
        required=False,
        type=int,
        default=40,
        help="training num epochs",
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
    parser.add_argument(
        "--initial-learning-rate",
        required=False,
        type=float,
        default=0.01,
        help="The initial learning rate of the model",
    )
    parser.add_argument(
        "--final-learning-rate",
        required=False,
        type=float,
        default=0.01,
        help="The final learning rate of the model",
    )

    args = parser.parse_args()
    
    training = TrainYOLO(args.base_model, args.experiments, args.name, args.eval_conf_th, args.yaml, args.project, args.neptune, args.batch_size, args.epochs, args.initial_learning_rate, args.final_learning_rate)
    training.train()
