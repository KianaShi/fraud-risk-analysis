"""Run gated Stage D freezing and Stage E final Test confirmation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection.foundation_finalization import (
    ensure_stage_e_outputs_available,
    persist_freeze_artifact,
    run_stage_e,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--test-only", action="store_true")
    parser.add_argument("--input", type=Path, default=root / "outputs" / "p2_dataset_clean.csv")
    parser.add_argument("--results-dir", type=Path, default=root / "docs" / "results")
    parser.add_argument("--figures-dir", type=Path, default=root / "docs" / "figures")
    parser.add_argument(
        "--tabpfn-checkpoint",
        type=Path,
        help="Local TabPFN checkpoint whose filename and SHA-256 match the Stage D freeze.",
    )
    parser.add_argument(
        "--tabicl-checkpoint",
        type=Path,
        help="Local TabICL checkpoint whose filename and SHA-256 match the Stage D freeze.",
    )
    parser.add_argument(
        "--allow-test-reproduction",
        action="store_true",
        help=(
            "Deliberate maintenance escape hatch permitting replacement of existing frozen "
            "Stage E outputs; this is not a normal model-development workflow."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation = args.results_dir / "foundation_model_validation.csv"
    stage_c_metadata = args.results_dir / "foundation_model_validation_metadata.json"
    freeze = args.results_dir / "foundation_model_configs.json"
    final_outputs = (
        args.results_dir / "foundation_model_test.csv",
        args.results_dir / "model_family_comparison.csv",
        args.figures_dir / "model_family_comparison.png",
    )
    if not args.freeze_only:
        ensure_stage_e_outputs_available(
            final_outputs, allow_test_reproduction=args.allow_test_reproduction
        )
    if not args.test_only:
        document = persist_freeze_artifact(validation, stage_c_metadata, freeze)
        statuses = {name: config["configuration_status"] for name, config in document["models"].items()}
        print(f"Stage D freeze persisted and reloaded successfully: {statuses}")
    if args.freeze_only:
        return
    if args.tabpfn_checkpoint is None or args.tabicl_checkpoint is None:
        raise SystemExit(
            "Stage E requires --tabpfn-checkpoint and --tabicl-checkpoint so their "
            "filenames and SHA-256 digests can be verified against the Stage D freeze."
        )
    # This is deliberately the first real-data read on the Stage E path, after
    # the freeze artifact has been persisted or independently revalidated.
    raw = pd.read_csv(args.input)
    test_results, comparison = run_stage_e(
        raw,
        freeze,
        validation,
        args.results_dir / "tuned_model_comparison.csv",
        args.results_dir / "final_tuned_test_comparison.csv",
        args.results_dir / "foundation_model_test.csv",
        args.results_dir / "model_family_comparison.csv",
        args.figures_dir / "model_family_comparison.png",
        checkpoint_paths={
            "tabpfn_3": args.tabpfn_checkpoint,
            "tabicl_v2": args.tabicl_checkpoint,
        },
        allow_test_reproduction=args.allow_test_reproduction,
    )
    print("\nStage E foundation-model Test results:")
    print(test_results.to_string(index=False))
    if not comparison.empty:
        print("\nFrozen four-model family comparison:")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
