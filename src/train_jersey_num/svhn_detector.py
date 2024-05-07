from abc import ABC, abstractmethod
from ultralytics import YOLO
import numpy as np

class NumberDetector(ABC):
    """
    Abstract base class for detecting numbers in images.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> bool:
        """
        Detects numbers in an RGB numpy array image.
        
        Parameters:
            image (numpy.ndarray): The image in which to detect numbers.
        
        Returns:
            List[int]: List of detected numbers.
        """
        pass
    
class SVHNNumberDet(NumberDetector):
    """
    Detector using a model trained on the SVHN dataset.
    """

    def __init__(self, model_path: str, threshold: float):
        """
        Initializes the detector with the specified pre-trained model.
        
        Parameters:
            model_path (str): Path to the pre-trained model file.
        """
        
        self.model = YOLO(model_path)
        self.threshold = threshold
        
    def detect(self, image: np.ndarray) -> bool:
        """
        Detects numbers in an image using the SVHN trained model.
        
        Parameters:
            image (numpy.ndarray): RGB image to detect numbers in.
        
        Returns:
            bool: True if any numbers are detected, False otherwise.
        """
        detections = self.model(image, verbose=False)
        
        # for det in detections:
        #     print(det.boxes)
        scores = [score for score in detections[0].boxes.conf.cpu().numpy().tolist() if score > self.threshold]
        return len(scores) > 0
