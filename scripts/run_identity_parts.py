from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism.identity_parts import main
if __name__=='__main__':main()
