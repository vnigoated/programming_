from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze
from .runner import expand_matrix, load_config, run_matrix


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


if __name__ == "__main__":
    main()

