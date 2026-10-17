from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze
from .artifacts import freeze_artifact, verify_manifest
from .readiness import primary_readiness, validate_primary_suite
from .primary_statistics import analyze_primary
from .runner import expand_matrix, load_config, run_matrix
from .study_analysis import analyze_study
from .validation import validate_artifact


def main() -> None:
    parser = argparse.ArgumentParser(prog="txnagentbench")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="Show the expanded run count without executing")
    plan.add_argument("--config", type=Path, required=True)

    run = commands.add_parser("run", help="Execute a restart-safe experiment matrix")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--max-runs", type=int)
    run.add_argument("--allow-scripted", action="store_true")

    report = commands.add_parser("analyze", help="Aggregate completed run summaries")
    report.add_argument("--output", type=Path, required=True)

    study = commands.add_parser(
        "analyze-study", help="Combine frozen blocks outside their immutable directories"
    )
    study.add_argument("--input", type=Path, action="append", required=True)
    study.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate", help="Validate a completed artifact against its frozen config")
    validate.add_argument("--output", type=Path, required=True)

    freeze = commands.add_parser("freeze", help="Validate and hash an immutable experiment artifact")
    freeze.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-manifest", help="Verify every file in a frozen artifact")
    verify.add_argument("--output", type=Path, required=True)

    readiness = commands.add_parser("readiness", help="Report publication-blocking primary-study gates")
    readiness.add_argument("--config", type=Path, required=True)
    readiness.add_argument("--output", type=Path)

    suite = commands.add_parser(
        "validate-primary-suite", help="Validate the complete split primary design"
    )
    suite.add_argument("--config", type=Path, action="append", required=True)

    primary_analysis = commands.add_parser(
        "analyze-primary", help="Run the frozen confirmatory primary analysis"
    )
    primary_analysis.add_argument("--input", type=Path, action="append", required=True)
    primary_analysis.add_argument("--output", type=Path, required=True)
    primary_analysis.add_argument("--plan", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "plan":
        config = load_config(args.config)
        rows = expand_matrix(config)
        print(json.dumps({"stage": config.get("stage"), "planned_runs": len(rows)}, indent=2))
    elif args.command == "run":
        result = run_matrix(args.config, args.output, args.max_runs, args.allow_scripted)
        print(json.dumps(result, indent=2))
    elif args.command == "analyze":
        print(json.dumps(analyze(args.output), indent=2))
    elif args.command == "analyze-study":
        print(json.dumps(analyze_study(args.input, args.output), indent=2))
    elif args.command == "validate":
        result = validate_artifact(args.output)
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
    elif args.command == "freeze":
        print(json.dumps(freeze_artifact(args.output), indent=2))
    elif args.command == "verify-manifest":
        result = verify_manifest(args.output)
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
    elif args.command == "readiness":
        result = primary_readiness(args.config, args.output)
        print(json.dumps(result, indent=2))
        if not result["ready"]:
            raise SystemExit(1)
    elif args.command == "validate-primary-suite":
        result = validate_primary_suite(args.config)
        print(json.dumps(result, indent=2))
        if not result["ready"]:
            raise SystemExit(1)
    elif args.command == "analyze-primary":
        result = analyze_primary(args.input, args.output, args.plan)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

