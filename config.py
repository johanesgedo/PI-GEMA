from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

TRANING_DATA = DATA_DIR / "training"

MICROTREMOR_DATA = DATA_DIR / "microtremor"

CALIBRATED_DATA = DATA_DIR / "calibrated"

PROCESSED_DATA = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CALIBRATED_OUTPUT = OUTPUTS_DIR / "calibrated"

FIGURES_OUTPUT = OUTPUTS_DIR / "figures"

REPORTS_OUTPUT = OUTPUTS_DIR / "reports"

TRANSFER_FUNCTION = (
    MODELS_DIR /
    "transfer_functions_streaming.npz"
)





