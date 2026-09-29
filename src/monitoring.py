from datetime import datetime, timezone
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "predictions.jsonl"

MODEL_VERSION = "1.0.0"


def log_prediction(features: dict, prediction: float) -> None:
    """
    Store inference data for model monitoring.

    JSON Lines is used so every prediction is stored as an
    independent record that can later be analyzed for drift,
    prediction distributions and model performance.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "features": features,
        "prediction": round(float(prediction), 2),
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")