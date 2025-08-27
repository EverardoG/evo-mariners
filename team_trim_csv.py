#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trim a fitness-like CSV to only include columns matching a pattern (default: team_<digits>_team_fitness) and always include a 'generation' column on the left if present. Writes to a 'trim_' prefixed CSV by default."
    )
    parser.add_argument(
        "input_path",
        help="Path to a CSV file or a directory containing fitness.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV file path (default: same directory as input, filename prefixed with 'trim_')",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--pattern",
        help="Regex pattern to select columns to keep (full match). Default: ^team_\\d+_team_fitness$",
        default=r"^team_\d+_team_fitness$",
    )
    parser.add_argument(
        "--encoding",
        help="CSV encoding (default: utf-8).",
        default="utf-8",
    )
    parser.add_argument(
        "--sep",
        help="CSV delimiter (default: ,).",
        default=",",
    )
    return parser.parse_args()


def resolve_paths(input_path: str, output: str | None) -> tuple[Path, Path]:
    in_path = Path(input_path)
    csv_path = in_path / "fitness.csv" if in_path.is_dir() else in_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV at: {csv_path}")

    # Default output is the same directory with 'trim_' prefixed to the input filename
    out_path = Path(output) if output else csv_path.with_name(f"trim_{csv_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, out_path


def main():
    args = parse_args()
    try:
        csv_path, out_path = resolve_paths(args.input_path, args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise SystemExit(1)

    try:
        df = pd.read_csv(csv_path, encoding=args.encoding, sep=args.sep)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        raise SystemExit(1)

    pattern = re.compile(args.pattern)
    keep_cols = [c for c in df.columns if pattern.fullmatch(c)]

    if not keep_cols:
        print(
            f"No columns matched pattern '{args.pattern}'. "
            f"Available columns: {', '.join(df.columns)}"
        )
        raise SystemExit(2)

    # Always include 'generation' column on the very left if present
    final_cols = keep_cols
    if "generation" in df.columns:
        final_cols = ["generation"] + [c for c in keep_cols if c != "generation"]

    trimmed = df[final_cols]

    try:
        trimmed.to_csv(out_path, index=False, encoding=args.encoding)
    except Exception as e:
        print(f"Failed to write output CSV: {e}")
        raise SystemExit(1)

    print(
        f"Wrote {len(final_cols)} columns and {len(trimmed)} rows to {out_path}"
    )


if __name__ == "__main__":
    main()
