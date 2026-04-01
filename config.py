"""
Configuration settings for Disaster Damage Detection system.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ─── Model Settings ─────────────────────────────────────────────────────────
IMAGE_SIZE = 224                 # Input image size for CNN
SEGMENTATION_SIZE = 256          # Input size for U-Net
NUM_CLASSES = 2                  # damaged / not-damaged
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda"                  # Will auto-fallback to "cpu"

# ─── Class Labels ────────────────────────────────────────────────────────────
CLASS_NAMES = ["No Damage", "Damaged"]

# ─── Color Palette ───────────────────────────────────────────────────────────
COLORS = {
    "no_damage": (46, 204, 113),    # Green
    "damaged":   (231, 76, 60),     # Red
    "overlay":   (231, 76, 60, 128) # Semi-transparent red
}

# ─── Map Settings ────────────────────────────────────────────────────────────
DEFAULT_LAT = 28.6139
DEFAULT_LON = 77.2090
DEFAULT_ZOOM = 12
