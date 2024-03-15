import os
import torch  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import logging
from dataclasses import dataclass
from typing import List, Any
from ultralytics import YOLO  # type: ignore
from tracklab.pipeline.imagelevel_module import ImageLevelModule  # type: ignore
from tracklab.utils.cv2 import cv2_load_image  # type: ignore
from tracklab.utils.coordinates import ltrb_to_ltwh  # type: ignore

os.environ["YOLO_VERBOSE"] = "False"
log = logging.getLogger(__name__)


@dataclass
class Box:
    xyxy: List[int]  # [left, top, right, bottom]
    conf: float
    cls: int


class FakeModel(torch.nn.Module):  # type: ignore
    def __init__(self, model_path: str) -> None:
        super(FakeModel, self).__init__()
        self.model_path = model_path
        self.conv1 = torch.nn.Conv2d(963, 128, kernel_size=3, padding=1)

    def forward(self, images: List[Any]) -> List[List[Box]]:
        fake_box1 = Box([200, 200, 400, 400], 0.85, 0)
        fake_box2 = Box([200, 500, 400, 700], 0.65, 0)
        fake_box3 = Box([800, 200, 400, 400], 0.3, 0)

        fake_boxes = [fake_box1, fake_box2, fake_box3]
        output = [fake_boxes for i, image in enumerate(images)]
        return output


def collate_fn(batch: List[Any]) -> tuple[List[Any], tuple[List[Any], List[Any]]]:
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class FakeYOLOv8(ImageLevelModule):  # type: ignore
    collate_fn = collate_fn
    input_columns: List[str] = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(
        self,
        cfg: dict[Any, Any],
        device: str,
        batch_size: int,
        **kwargs: dict[str, Any]
    ) -> None:
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.model = FakeModel(cfg.path_to_checkpoint)
        self.model.to(device)
        self.id = 0

    @torch.no_grad()
    def preprocess(
        self,
        image: np.ndarray,
        detections: pd.DataFrame,
        metadata: pd.Series,
    ) -> dict[str, Any]:
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(
        self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame
    ) -> pd.DataFrame:
        images, shapes = batch
        results_by_image = self.model(images)
        detections = []
        for image_results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in image_results:
                # check for `person` class
                if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence:
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(np.array(bbox.xyxy), shape),
                                bbox_conf=bbox.conf,
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections
