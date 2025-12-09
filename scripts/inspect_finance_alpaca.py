#!/usr/bin/env python3
"""Inspect and persist the Finance-Alpaca dataset locally."""

import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Finance-Alpaca and save it locally.")
    parser.add_argument(
        "--dataset",
        default="gbharti/finance-alpaca",
        help="Dataset identifier on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_raw/finance_alpaca",
        help="Directory where the dataset will be saved via `save_to_disk`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds = load_dataset(args.dataset)
    print(ds)
    print(ds["train"][0])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_dir))
    print(f"Dataset saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
