"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import AnalysisConfig
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze a multi-position ND2 gradient time course.")
    parser.add_argument("nd2_file", type=Path, help="Input Nikon .nd2 file")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Parent output folder")
    parser.add_argument("--config", type=Path, help="Optional JSON configuration file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.nd2_file.is_file():
        raise SystemExit(f"ND2 file not found: {args.nd2_file}")
    # Keep heavy image/plotting imports out of argument parsing, so --help works
    # before the project dependencies have been installed.
    from .pipeline import run_pipeline

    config = AnalysisConfig.from_json(args.config)
    destination = run_pipeline(args.nd2_file, args.output, config)
    print(f"Analysis complete: {destination.resolve()}")
    return 0
