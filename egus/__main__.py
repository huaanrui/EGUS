"""Public commands. Heavy GPU dependencies are imported only when needed."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(prog="python -m egus")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare-data", help="Restore the exact hash-checked BBQ partition")
    data.add_argument("--raw-dir")
    data.add_argument("--data-dir")
    data.add_argument("--offline", action="store_true")
    download = sub.add_parser("download-models", help="Download pinned checkpoints")
    download.add_argument("--models", nargs="+")
    download.add_argument("--models-root", default="models")
    download.add_argument("--endpoint", default="https://hf-mirror.com")
    sub.add_parser("fetch-sources", help="Fetch and verify pinned author source files")
    verify = sub.add_parser(
        "verify", help="Recompute released tables on CPU without loading a model"
    )
    verify.add_argument("--statistics", action="store_true")
    verify.add_argument("--output")
    p = sub.add_parser("run", help="Run the validation-selected paper experiments")
    p.add_argument("--config", default="configs/paper.toml")
    p.add_argument("--models", nargs="+")
    p.add_argument("--models-root")
    p.add_argument("--output")
    p.add_argument("--stage", choices=["all", "fit", "evaluate"], default="all")
    args = parser.parse_args()
    if args.command == "prepare-data":
        from .data import prepare

        result = prepare(args.raw_dir, args.data_dir, args.offline)
    elif args.command == "verify":
        from .verify import verify

        result = verify(args.statistics, args.output)
    elif args.command == "fetch-sources":
        from .baselines.author import fetch_sources

        result = fetch_sources()
    elif args.command == "download-models":
        from .download import download

        result = download(args.models_root, args.models, args.endpoint)
    else:
        from .config import load

        config = load(args.config, args.models, args.output, args.models_root)
        from .experiment import run

        result = run(config, args.stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
