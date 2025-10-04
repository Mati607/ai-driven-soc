#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from app.reports.weekly import generate_weekly_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weekly SOC HTML report from triage JSONL")
    parser.add_argument("triage_jsonl", type=Path, help="Path to triage_results.jsonl")
    parser.add_argument("--out", type=Path, default=Path("reports/weekly_report.html"))
    args = parser.parse_args()

    generate_weekly_report(args.triage_jsonl, args.out)
    print(f"Wrote report to {args.out}")


if __name__ == "__main__":
    main()


