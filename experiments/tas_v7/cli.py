"""CPU preparation, GPU study and CPU reporting entry point used by Colab."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from tas_runtime.config import StudyConfig


def plan(config):
    count = len(config.methods) or 14
    seeds = len(config.seeds)
    methods_confirm = 5 + config.shortlist_count
    return {
        "initial_methods": count,
        "smoke": 30 * count,
        "screen_initial": 240 * count,
        "screen_protocol_controls": 240 * 2,
        "confirmation_sample_and_greedy": 300 * methods_confirm * (seeds + 1)
        if config.run_confirmation
        else 0,
        "controls": "240 images, one seed plus greedy; fixed controls and temperature baselines, identity-separated",
        "diagnostics": f"{config.diagnostic_count} images, E0/E1/one finalist, D-PRED/D-MASK/D-BOX/D-MISS; fixed-prefix probes",
        "budget_minutes_per_session": config.max_session_minutes,
        "resume": "completed image/method/seed/config records are reused across sessions",
        "stage_flow": "automatic after blocking P0; smoke review pause overridden by owner",
        "shortlist": {
            "coverage_baseline": "E1-H",
            "maximum_coverage_loss_percentage_points": 100 * config.coverage_tolerance,
            "eligibility": "observed CHAIRi or CHAIRs improvement and no Pareto domination by matched sampled policies",
            "no_eligible_candidate": "save report and stop dependent stages",
        },
        "not_automatic": [
            "opening final remainder",
            "P3",
            "P4",
            "P5",
            "new backbone",
            "FG-G",
        ],
        "runtime": "not predicted before a complete A100 smoke; no session-completion promise",
    }


def score_saved(config):
    from .report import write_report
    from tas_runtime.artifacts import canonical_hash

    root = Path(config.persistent_root).expanduser() / "runs"
    from tas_runtime.persistence import read_incremental_snapshot

    outputs = []
    for pointer in sorted(root.glob("*/**/snapshots/ledger-index.json")):
        value = read_incremental_snapshot(pointer)
        if canonical_hash(value["identity"]) != value["identity_hash"]:
            raise RuntimeError(f"Snapshot identity mismatch: {pointer}")
        latest = {}
        for row in value["records"]:
            latest[row["key_hash"]] = row
        records = [row["payload"] for row in latest.values() if row["status"] == "ok"]
        failures = [row for row in latest.values() if row["status"] != "ok"]
        target = pointer.parent.parent
        identity = value["identity"]
        expected = (
            len(identity["rows"])
            * len(identity["methods"])
            * sum(
                len(identity["seeds"]) if mode == "sample" else 1
                for mode in identity["modes"]
            )
        )
        previous_path = target / "report.json"
        metadata = (
            json.loads(previous_path.read_text()).get("metadata", {})
            if previous_path.exists()
            else {}
        )
        metadata.update(
            identity=identity,
            reconstructed_from_snapshot=True,
            complete=len(records) == expected and not failures,
        )
        write_report(
            target,
            records,
            failures=failures,
            expected=expected,
            metadata=metadata,
            bootstrap_repeats=config.bootstrap_replicates,
            bootstrap_seed=config.bootstrap_seed,
        )
        outputs.append(str(target / "report.json"))
    return {"reports": outputs, "count": len(outputs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase", choices=("prepare", "run", "score", "check", "plan"), required=True
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Data-only CPU preparation; never used for GPU run.",
    )
    args = parser.parse_args(argv)
    config = StudyConfig.load(args.config)
    if os.environ.get("PYTHONHASHSEED") != "0" and args.phase == "run":
        raise RuntimeError(
            "Set PYTHONHASHSEED=0 before starting Python; inventory label order must be reproducible across sessions."
        )
    if args.phase == "plan":
        output = plan(config)
    elif args.phase == "prepare":
        from .study import prepare

        cohorts = (
            ("A", "B", "C", "D", "remaining")
            if config.run_final
            else ("A", "B", "C", "D")
        )
        output = prepare(config, models=not args.skip_models, cohorts=cohorts)
    elif args.phase == "run":
        from .study import Study

        output = Study(config).run()
    elif args.phase == "score":
        output = score_saved(config)
    else:
        # No weights/network required; invokes the scientific and miniature-model suite.
        root = Path(__file__).resolve().parents[2]
        subprocess.run(
            [sys.executable, "-m", "pytest", str(root / "tests"), "-q"],
            cwd=root,
            check=True,
        )
        output = {"checks": "passed"}
    print(json.dumps(output, indent=2), flush=True)
    return output


if __name__ == "__main__":
    main()
