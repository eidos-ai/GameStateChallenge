from typing import Any, List
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from omegaconf import OmegaConf
from yacs.config import CfgNode as CN

from prtreid.scripts.main import build_config, build_torchreid_model_engine
from prtreid.tools.feature_extractor import FeatureExtractor
from prtreid.utils.imagetools import build_gaussian_heatmaps

import prtreid
from prtreid.utils.tools import extract_test_embeddings

def inspect(name, obj, depth=0):
    indent = '|-' * depth
    indent_next = '|--' * (depth + 1)
    if isinstance(obj, dict):
        print(f"{indent}{name}: {type(obj)}")
        for k, v in obj.items():
            print(f"{indent_next}{k}:")
            inspect('', v, depth + 1)
    elif isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
        print(f"{indent}{name}: {obj.shape}")
    elif isinstance(obj, tuple):
        print(f"{indent}{name}: {type(obj)}")
        for i, v in enumerate(obj):
            print(f"{indent_next}{i}:")
            inspect('', v, depth + 1)
    else:
        print(f"{indent}{name}: {type(obj)}")

from torch import nn
    
class BNClassifier(nn.Module):
    # Source: https://github.com/upgirlnana/Pytorch-Person-REID-Baseline-Bag-of-Tricks
    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)  # BoF: this doesn't have a big impact on perf according to author on github
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self._init_params()

    def forward(self, x):
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
class PRTReId(torch.nn.Module):
    def __init__(self, embedding_dim: int, model_path: str, hrnet_dir: str, config_path: str,
                device: str, target_embeddings="globl"):
        """
        target_embeddings can be one of 'globl', 'foreg', 'conct', 'parts'
        """
        super(PRTReId, self).__init__()
        
        self.model_path = model_path
        self.hrnet_dir = hrnet_dir
        self.config_path = config_path
        self.target_embeddings=target_embeddings
        
        cfg = OmegaConf.load(config_path)
        cfg = cfg.cfg
        del cfg.data
        del cfg.project
        del cfg.sampler
        del cfg.model.bpbreid.masks
        cfg.model.load_weights = model_path
        cfg.model.bpbreid.hrnet_pretrained_path = hrnet_dir
        cfg = CN(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = build_config(config=cfg)

        self.feature_extractor = FeatureExtractor(
                    self.cfg,
                    model_path=cfg.model.load_weights,
                    device=device,
                    image_size=(256, 128),
                    model=None,
                    verbose=False,  # FIXME @Vladimir
                )
        self.global_team_head = BNClassifier(256, embedding_dim)
        
        self.preprocess = self.feature_extractor.preprocess
        
    def get_embeddings(self, model_input: Any):
        prtreid_result = self.feature_extractor(model_input, external_parts_masks=None)
        embeddings, vis_scores, embeddings_masks, pixels_cls_scores, role_cls_score = extract_test_embeddings(prtreid_result, [self.target_embeddings])
        
        return embeddings.squeeze(1)
    
    def forward(self, model_input_tensor: torch.Tensor):
        model_output = self.feature_extractor.model(model_input_tensor, external_parts_masks=None)
        backbone_embeddings, visibility_scores, id_cls_scores, team_cls_score, role_cls_score, pixels_cls_scores, spatial_features, parts_masks = model_output
        # inspect("model", self.feature_extractor.model )
        # inspect("backbone_embeddings", backbone_embeddings)
        # inspect("visibility_scores", visibility_scores)
        # inspect("id_cls_scores", id_cls_scores)
        # inspect("team_cls_score", team_cls_score)
        # inspect("role_cls_score", role_cls_score)
        # inspect("pixels_cls_scores", pixels_cls_scores)
        # inspect("spatial_features", spatial_features)
        # inspect("parts_masks", parts_masks)
        
        backbone_embeddings = backbone_embeddings[self.target_embeddings]
        if self.target_embeddings == "parts":
            backbone_embeddings = backbone_embeddings.squeeze(dim = 1)
        _, team_embeddings = self.global_team_head(backbone_embeddings)
        return team_embeddings
    
    def preprocess_pil_image(self, pil_image: Image):
        image_array = np.array(pil_image)
        
        preprocess_input = {"image":image_array}
        output = self.feature_extractor.preprocess(**preprocess_input)
        
        return output["image"]
        
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--model', required=True, help='Path to trained prtreid model')
    parser.add_argument('-hr', '--hrnet_pretrained', required=True, help='Path to pre trained hrnet model')
    parser.add_argument('-cfg', '--config', required=False, help='Path to prtreid config yaml file')
    
    args = vars(parser.parse_args())
    
    # Get cpu, gpu or mps device for training.
    device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
    print(f"Using {device} device")
    
    model_path = args["model"]
    config_path = args["config"]
    hrnet_dir = args["hrnet_pretrained"]
    
    prtreid = PRTReId(6, model_path, hrnet_dir, config_path, device, 'globl')

    sample_batch_images = ["sample_crop.png", "sample_crop.png"]
    model_input_pils = [Image.open(img_path).convert("RGB") for img_path in sample_batch_images] 
    preprocessed_inputs = [prtreid.preprocess_pil_image(input_pil).unsqueeze(0) for input_pil in model_input_pils]
    preprocessed_inputs = torch.cat(preprocessed_inputs)
    preprocessed_inputs = preprocessed_inputs.to(device)
    
    # This section simulates getting embeddings when training model
    prtreid.train()
    embeddings_with_grad = prtreid(preprocessed_inputs)

    # This section simulates getting embeddings when evaluating model
    prtreid.eval()
    with torch.no_grad():
        embeddings = prtreid(preprocessed_inputs)
        print(embeddings[0])
    prtreid.train()
    torch.save(prtreid, "../../pretrained_models/reid/eidos_prtreid_base.pt")
    # This section simulates loading the model and getting embeddings. Printed tensor should be the same as last section
    loaded_prtreid = torch.load("../../pretrained_models/reid/eidos_prtreid_base.pt")
    loaded_prtreid.eval()
    with torch.no_grad():
        embeddings_loaded_model = loaded_prtreid(preprocessed_inputs)
        print(embeddings_loaded_model[0])