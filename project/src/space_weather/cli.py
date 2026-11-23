import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze space weather launch risk"
    )
    parser.add_argument("--start-date", default="2025-05-01")
    parser.add_argument("--end-date", default="2025-07-09")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run without displaying matplotlib windows",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_directory = Path(__file__).resolve().parents[2]
    run_pipeline(
        project_directory,
        start_date=args.start_date,
        end_date=args.end_date,
        show_dashboard=not args.no_show,
    )
    return 0
