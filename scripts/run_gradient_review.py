"""Run the chronological gradient review (corrections, calibration, bridge, stain, Z selection, fits).

September full experiment (main result at the config Z, Z11; robustness check = mean of Z3-Z12).
``--reuse`` points at earlier cache folders so already-measured timepoints are not re-read:

    .\\.venv\\Scripts\\python.exe scripts\\run_gradient_review.py ^
        "D:\\Users\\Shelly\\2026_09_27_Exp106_try4\\gradient.nd2" ^
        --config exp106_2026_09_27_full_experiment_config.json ^
        --label 2026_09_27_exp106_try4 --layer 3 12 --z-timepoints 0 6 12 18 24 ^
        --output outputs\\2026_09_27_exp106_try4_full_experiment ^
        --reuse <earlier run>\\cache_measurements_do_not_edit

July example (no calibration, DIC, or stain; Z selection at the July all-Z timepoints):

    .\\.venv\\Scripts\\python.exe scripts\\run_gradient_review.py ^
        "D:\\Users\\Shelly\\2026_07_30_exp106_grad_2C\\2026_07_30_gradient.nd2" ^
        --config exp106_2026_07_30_gradient_review_config.json ^
        --label 2026_07_30_exp106 --no-stain-mask --z-timepoints 2 10 14 24 36 ^
        --output outputs\\2026_07_30_exp106_grad_2C_gradient_review

Every invocation writes a new timestamped run directory. ``--reuse`` folders are read only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradient_analysis.config import AnalysisConfig  # noqa: E402
from gradient_analysis.gradient_review import run_gradient_review  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nd2_path")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True, help="Experiment label used at the start of every file name.")
    parser.add_argument("--workers", type=int, default=48, help="Worker processes (Windows maximum 61).")
    parser.add_argument("--bin-factor", type=int, default=8)
    parser.add_argument("--reuse", nargs="*", default=[], help="Earlier cache folders to reuse (read only).")
    parser.add_argument("--z-timepoints", nargs="*", type=int, default=None,
                        help="Zero-based timepoints for the Z-selection step (default: all analysed timepoints).")
    parser.add_argument("--no-stain-mask", action="store_true")
    parser.add_argument("--z-first", type=int, default=1, help="First one-based Z plane to analyse (e.g. 4 to skip planes in the glass).")
    parser.add_argument("--glass-plane", type=int, default=None, help="One-based Z plane at the glass surface (for height labels).")
    parser.add_argument("--no-flagged-fit-analysis", action="store_true")
    parser.add_argument("--diffusion", action="store_true", help="Add the front displacement / D(t) step.")
    parser.add_argument("--expected-d", nargs=2, type=float, default=None, metavar=("LOW", "HIGH"),
                        help="Expected diffusion coefficient range [µm²/s] shown on the D(t) plot.")
    parser.add_argument("--layer", nargs=2, type=int, default=None, metavar=("FIRST_Z", "LAST_Z"),
                        help="One-based Z planes averaged for the robustness check, e.g. --layer 3 12.")
    args = parser.parse_args()
    run_dir = run_gradient_review(
        args.nd2_path,
        args.output,
        AnalysisConfig.from_json(args.config),
        experiment_label=args.label,
        workers=min(args.workers, 61),
        bin_factor=args.bin_factor,
        reuse_dirs=[Path(item) for item in args.reuse],
        z_analysis_timepoints=tuple(args.z_timepoints) if args.z_timepoints else None,
        stain_mask_enabled=not args.no_stain_mask,
        layer=tuple(args.layer) if args.layer else None,
        z_first_one_based=args.z_first,
        glass_plane_one_based=args.glass_plane,
        flagged_fit_analysis=not args.no_flagged_fit_analysis,
        diffusion_analysis=args.diffusion,
        expected_d=tuple(args.expected_d) if args.expected_d else None,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
