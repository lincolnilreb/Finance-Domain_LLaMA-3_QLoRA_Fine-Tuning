#!/usr/bin/env python3
"""Concatenate multiple ChatML JSONL files into a single output."""

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ChatML jsonl shards.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files (order determines concatenation unless --shuffle is used).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination file path.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle all rows before writing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is specified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = []
    for path_str in args.inputs:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            buffer.extend(line.rstrip("\n") for line in file if line.strip())

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(buffer)

    with output_path.open("w", encoding="utf-8") as out_file:
        for line in buffer:
            out_file.write(line + "\n")

    print(f"Wrote {len(buffer)} records to {output_path.resolve()}.")


if __name__ == "__main__":
    main()
