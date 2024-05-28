import pandas as pd
import numpy as np
import torch
import logging
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from models.resnet import ResnetMLP
from models.resnet import TRANSFORM
from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule
from gsr.inspector import inn
from torchvision import transforms


log = logging.getLogger(__name__)

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class RoleEmbeddings(DetectionLevelModule):
    input_columns = []
    output_columns = ["role_embeddings"]
    collate_fn = default_collate

    def __init__(self, cfg, device, batch_size, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.cfg = cfg
        self.model = ResnetMLP(
            resnet_model="resnet18",
            mlp_hidden_layers=[],
            mlp_output_size=64,
        )
        self.model.load_state_dict(torch.load(self.cfg.model_path))
        self.model.to(self.cfg.device)
        self.model.eval()

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        batch = ["img"]
        l, t, r, b = detection.bbox.ltrb(image_shape=(image.shape[1], image.shape[0]), rounded=True)
        crop = image[t:b, l:r]
        crop_transformed = TRANSFORM(crop)
        batch = {"img": crop_transformed}
        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):       
        with torch.no_grad():
            img = batch["img"].to(self.cfg.device)
            embeddings = self.model(img)
            embeddings_array = embeddings.cpu().numpy()
        array_list = [np.array(row) for row in embeddings_array]    
        detections['role_embeddings'] = array_list

        return detections