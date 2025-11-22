import json
import logging
import os
import warnings
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = Path(os.path.join(ROOT_DIR, "models"))
if not os.path.exists(MODELS_DIR):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

SECRETS_DIR = Path(os.path.join(ROOT_DIR, "secrets"))
if not os.path.exists(SECRETS_DIR):
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)

MAP_FILE = os.path.join(ROOT_DIR, "configs.json")
if not os.path.exists(MAP_FILE):
    raise FileNotFoundError(f"❌ config JSON not found: {MAP_FILE}")


def load_path_map():
    """Đọc file JSON map: tên model PATH → {file_id, filename, is_zip}."""
    with open(MAP_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


class Paths:
    """Quản lý toàn bộ đường dẫn và ánh xạ model theo JSON cấu hình."""

    # Các key tương ứng với loại model
    MODEL_KEYS = [
        "YOLOv8N-Face",
        "EFFICIENTNET-B3",
    ]

    # Nạp JSON map khi khởi tạo
    def __init__(self):
        self.path_map = load_path_map()
        self.paths = self._get_model_paths()

    def _get_model_paths(self):
        """Tạo dict ánh xạ model_key → thông tin đầy đủ (path, filename, file_id, is_zip)."""
        paths = {}

        for key in self.MODEL_KEYS:
            info = self.path_map.get(key)

            if info:
                filename = info.get("filename", "")
                full_path = os.path.join(MODELS_DIR, filename)
                paths[key] = {
                    "full_path": full_path,
                    "base_path": MODELS_DIR,
                    "filename": filename,
                    "file_id": info.get("file_id"),
                    "is_zip": info.get("is_zip", False),
                }
            else:
                # Không có trong JSON → chỉ lưu base path
                paths[key] = {
                    "full_path": MODELS_DIR,
                    "base_path": MODELS_DIR,
                    "filename": None,
                    "file_id": None,
                    "is_zip": None,
                }
                logging.warning(f"⚠️  No entry found for {key} in {MAP_FILE}")

        return paths
