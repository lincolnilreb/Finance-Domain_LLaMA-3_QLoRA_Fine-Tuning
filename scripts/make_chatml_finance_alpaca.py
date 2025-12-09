#!/usr/bin/env python3
"""Convert Finance-Alpaca into ChatML-ready JSONL files."""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

DEFAULT_MAX_SAMPLES = 10_000
DEFAULT_VAL_RATIO = 0.05
DEFAULT_SYSTEM_PROMPT = "You are a helpful, precise financial assistant."


def build_user_content(example) -> str:
    instr = example["instruction"].strip()
    inp = (example.get("input") or "").strip()
    if inp:
        return instr + "\n\nContext:\n" + inp
    return instr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ChatML jsonl files from Finance-Alpaca.",
    )
    parser.add_argument(
        "--dataset",
        default="gbharti/finance-alpaca",
        help="HF hub identifier or local path saved via `save_to_disk`.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to convert (ignored for simple datasets saved locally).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Cap number of samples (set <=0 or omit to keep all).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Fraction of examples reserved for validation (0 < ratio < 1).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System message included in every ChatML example.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_processed",
        help="Directory that will receive the jsonl files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for shuffling prior to sampling.",
    )
    return parser.parse_args()


def load_split(dataset: str, split: str) -> Dataset:
    dataset_path = Path(dataset)
    if dataset_path.exists():
        loaded = load_from_disk(str(dataset_path))
        if isinstance(loaded, DatasetDict):
            if split not in loaded:
                raise ValueError(f"Split '{split}' not found in dataset at {dataset_path}.")
            return loaded[split]
        return loaded
    ds = load_dataset(dataset, split=split)
    if isinstance(ds, DatasetDict):
        return ds[split]
    return ds


def main() -> None:
    args = parse_args()

    if not 0 < args.val_ratio < 1:
        raise ValueError("`--val-ratio` must be strictly between 0 and 1.")

    dataset = load_split(args.dataset, args.split)
    dataset = dataset.shuffle(seed=args.seed)

    if args.max_samples and args.max_samples > 0 and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))

    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    if n_val >= n_total:
        raise ValueError(
            "Validation split would consume the entire dataset. Reduce `--val-ratio`.",
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_finance_alpaca_chatml.jsonl"
    val_path = out_dir / "val_finance_alpaca_chatml.jsonl"

    def example_to_chatml(example):
        user_content = build_user_content(example)
        assistant_content = example["output"].strip()
        return {
            "messages": [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        }

    split_idx = n_total - n_val
    with train_path.open("w", encoding="utf-8") as f_train, val_path.open(
        "w",
        encoding="utf-8",
    ) as f_val:
        for i, ex in enumerate(dataset):
            chatml = example_to_chatml(ex)
            line = json.dumps(chatml, ensure_ascii=False)
            if i < split_idx:
                f_train.write(line + "\n")
            else:
                f_val.write(line + "\n")

    print(f"Saved {split_idx} train and {n_val} val examples.")
    print("Train:", train_path.resolve())
    print("Val:", val_path.resolve())


if __name__ == "__main__":
    main()
