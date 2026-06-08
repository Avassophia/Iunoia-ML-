import json
from pathlib import Path

def load_constants():
    BASE_DIR = Path(__file__).resolve().parents[1]
    CONST_PATH = BASE_DIR / "config" / "constants.json"

    with open(CONST_PATH) as f:
        return json.load(f)