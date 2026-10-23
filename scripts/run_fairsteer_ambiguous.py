"""Run only the ambiguous-direction protocol correction, with frozen diagnostics."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mechanism.fairsteer import FairSteerExperiment
from mechanism.fairsteer_data import prepare
from mechanism.fairsteer_revision import prepare_confirmation,freeze_revision,additional_tests


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",nargs="?",default="all",choices=("prepare","all","additional","report"));a=p.parse_args()
    config=ROOT/"configs/experiment_fairsteer_ambiguous.toml"
    prepare(config);manifest=prepare_confirmation(config)
    print({"confirmation_prepared":manifest["rows"],"categories":manifest["categories"]},flush=True)
    if a.stage=="prepare":return
    e=FairSteerExperiment(config)
    freeze_revision(e)
    if a.stage=="all":
        for stage in ("fit","checks","screen","select","test","stability","report"):
            if not e.state["stages"].get(stage,{}).get("completed"):getattr(e,stage)()
    if a.stage in ("all","additional") and not e.state["stages"].get("additional_tests",{}).get("completed"):
        if not e.state["stages"].get("test",{}).get("completed"):raise ValueError("Complete the main diagnostic first")
        additional_tests(e)
    if a.stage in ("all","report"):
        from mechanism.fairsteer_revision_report import render_revision
        render_revision(e.out,e.data,ROOT/e.c["confirmation_dir"])


if __name__=="__main__":main()
