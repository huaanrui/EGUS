from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mechanism.fairsteer_data import prepare

if __name__ == "__main__":
    manifest = prepare(ROOT / "configs/experiment_fairsteer.toml")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
