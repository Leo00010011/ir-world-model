"""DINO-DETR head on frozen DINOv2-Reg features (F7, spec/2026-10-23-dino-detr-head-port)."""

from irwm.detection.detector import DINODetector, FeatureDINO, build_dino_config
from irwm.detection.detrex_adapter import cuda_ops_available
from irwm.detection.stem import FeatureStem

__all__ = ["DINODetector", "FeatureDINO", "FeatureStem", "build_dino_config", "cuda_ops_available"]
