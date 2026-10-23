from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mechanism.fairsteer import main

if __name__ == "__main__":
    if len(sys.argv)==1:
        sys.argv.append("all")
    main()
