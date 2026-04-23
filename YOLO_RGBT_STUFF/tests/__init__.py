# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from YOLO_RGBT_STUFF.ultralytics.utils import ASSETS, ROOT, WEIGHTS_DIR
from YOLO_RGBT_STUFF.ultralytics.utils import checks

# Constants used in tests
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo11n.pt"  # test spaces in path
CFG = "yolo11n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]
TMP = (ROOT / "../tests/tmp").resolve()  # temp directory for test files
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()

__all__ = (
    "MODEL",
    "CFG",
    "SOURCE",
    "SOURCES_LIST",
    "TMP",
    "CUDA_IS_AVAILABLE",
    "CUDA_DEVICE_COUNT",
)
