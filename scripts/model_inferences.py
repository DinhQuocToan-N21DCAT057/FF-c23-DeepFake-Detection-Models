import logging
import threading
import warnings
import torch
import timm
import torch.nn as nn

from ultralytics import YOLO
from model_downloader import ModelDownloader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = ModelDownloader()


class YOLOv8NInference:
    """YOLOv8N Face Detector"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(YOLOv8NInference, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    @classmethod
    def _load_model(cls):
        """Load YOLOv8N-Face model từ file config."""
        loader.download_single("YOLOv8N-Face")
        model_path = loader.cfg.paths["YOLOv8N-Face"]["full_path"]
        model = YOLO(model_path)
        logging.info(f"✅ YOLOv8N-Face model loaded successfully from {model_path}")
        return model

    @classmethod
    def _get_model(cls, reload=False):
        """Lấy instance YOLOv8N-Face (cache)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, frame):
        result = self._get_model().predict(frame, conf=0.5, imgsz=640, verbose=False)
        return result


class EFFNetB3Inference(nn.Module):
    """EfficientNet-B3"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EFFNetB3Inference, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name='efficientnet_b3', is_pretrained=False, num_classes=0):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=is_pretrained, num_classes=num_classes)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.backbone.num_features, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.classifier(x)

    @classmethod
    def _load_model(cls):
        """Load EFFICIENTNET-B3 weight từ file config."""
        loader.download_single("EFFICIENTNET-B3")
        weights_path = loader.cfg.paths["EFFICIENTNET-B3"]["full_path"]
        model = EFFNetB3Inference().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        logging.info(f"✅ EFFICIENTNET-B3 model & weight loaded successfully from {weights_path}")
        return model

    @classmethod
    def _get_model(cls, reload=False):
        """Lấy instance EFFICIENTNET-B3 (cache)."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, face_tensor):
        with torch.no_grad():
            proba = torch.softmax(self._get_model()(face_tensor), dim=1)[0, 1].item()
            label = "FAKE" if proba > 0.5 else "REAL"
            conf = proba if label == "FAKE" else 1 - proba

        return label, conf
