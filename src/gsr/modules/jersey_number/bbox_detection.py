import pandas as pd
import numpy as np
import torch
import logging
import sys
import os

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule
from torchvision import transforms

from gsr.modules.jersey_number.models.convnext import ConvNext

log = logging.getLogger(__name__)

    
class BBoxJerseyNumberDetection(DetectionLevelModule):
    input_columns = []
    output_columns = ["bbox_jersey_number"]
    collate_fn = default_collate
    
    def __init__(self, cfg, device, batch_size, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.cfg = cfg
        learning_rate = 1e-4 # Not used really
        self.model = ConvNext(self.cfg.model_name, 100, learning_rate)
        self.model.load_model(self.cfg.model_path)
        self.model.to(self.cfg.device)
        self.model.eval()
        self.inference_transform = self.model.get_transform(is_training= False)
        
    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        batch = ["img"]
        l, t, r, b = detection.bbox.ltrb(image_shape=(image.shape[1], image.shape[0]), rounded=True)
        crop = image[t:b, l:r]
        crop_transformed = self.inference_transform(crop)
        batch = {"img": crop_transformed}
        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):       
        with torch.no_grad():
            img = batch["img"].to(self.cfg.device)
            pred_scores = self.model(img)
            pred_scores_array = pred_scores.cpu().numpy()
        array_list = [np.array(row) for row in pred_scores_array]    
        detections['jersey_number_detection'] = array_list
        return detections
