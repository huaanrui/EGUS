"""Run/resume the second pilot without overwriting the first pilot."""
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    subprocess.run([sys.executable, '-u', 'scripts/prepare_round2.py'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-u', 'scripts/calibration_round2.py', 'preregister'], cwd=ROOT, check=True)
    helper = subprocess.Popen([sys.executable, '-u', 'scripts/calibration_round2.py', 'all'], cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        subprocess.run([sys.executable, '-u', '-m', 'mechanism.round2', 'all'], cwd=ROOT, check=True)
        if helper.wait() != 0:
            raise RuntimeError('Calibration control failed')
        subprocess.run([sys.executable, '-u', 'scripts/summarize_round2_ablation.py'], cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-u', 'scripts/finalize_round2.py'], cwd=ROOT, check=True)
        subprocess.run([sys.executable, '-u', 'scripts/audit_round2.py'], cwd=ROOT, check=True)
    finally:
        if helper.poll() is None:
            helper.terminate()
