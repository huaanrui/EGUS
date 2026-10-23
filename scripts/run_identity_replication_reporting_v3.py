"""Replace only the saved-record auditor for old frozen scientific runners."""
from pathlib import Path
import sys,runpy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism.identity_replication import Experiment
from scripts.audit_identity_replication_records_v3 import audit

if __name__=='__main__':
    target=sys.argv[1];sys.argv=sys.argv[1:]
    Experiment.audit=audit
    runpy.run_path(target,run_name='__main__')
