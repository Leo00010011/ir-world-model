"""DINO-DETR head on frozen DINOv2-Reg features (F7, spec/2026-10-23-dino-detr-head-port).

The frozen heads ``det_rgb``/``det_ir`` (F8, spec/2026-10-26-detector-training-pipeline) load with
``load_frozen_detector``. The training module lives in ``irwm.detection.lightning``, not imported here.
"""

from irwm.detection.detector import DINODetector, FeatureDINO, build_dino_config
from irwm.detection.detrex_adapter import cuda_ops_available
from irwm.detection.frozen import frozen_meta, load_frozen_detector
from irwm.detection.stem import FeatureStem

__all__ = ["DINODetector", "FeatureDINO", "FeatureStem", "build_dino_config", "cuda_ops_available",
           "frozen_meta", "load_frozen_detector"]
