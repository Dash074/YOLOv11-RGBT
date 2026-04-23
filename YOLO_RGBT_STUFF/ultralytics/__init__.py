# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.75"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from YOLO_RGBT_STUFF.ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld
from YOLO_RGBT_STUFF.ultralytics.utils import ASSETS, SETTINGS
from YOLO_RGBT_STUFF.ultralytics.utils.checks import check_yolo as checks
from YOLO_RGBT_STUFF.ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
)
