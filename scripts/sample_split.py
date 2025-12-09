#!/usr/bin/env python3
"""Sample reproducible train/eval subsets from a dataset split."""

import argparse
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample train/eval subsets from an existing dataset split.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name on the Hugging Face Hub or a local path saved via `save_to_disk`.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Which split to sample from (ignored for single-split datasets saved locally).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=68912,
        help="Total number of examples to draw from the split.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=1000,
        help="How many examples to reserve for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_processed/sampling_v1",
        help="Directory where sampled splits will be written.",
    )
    parser.add_argument(
        "--train-name",
        default="train_sample",
        help="Folder name for the sampled train split.",
    )
    parser.add_argument(
        "--eval-name",
        default="eval_sample",
        help="Folder name for the sampled eval split.",
    )
    return parser.parse_args()


def load_split(dataset: str, split: str):
    """Load a dataset split from hub or local disk."""
    dataset_path = Path(dataset)
    if dataset_path.exists():
        loaded = load_from_disk(str(dataset_path))
        if isinstance(loaded, DatasetDict):
            if split not in loaded:
                raise ValueError(f"Split '{split}' not found in dataset at {dataset_path}.")
            return loaded[split]
        return loaded
    return load_dataset(dataset, split=split)


def main() -> None:
    args = parse_args()

    if args.eval_size >= args.num_examples:
        raise ValueError("`eval_size` must be smaller than `num_examples`.")

    dataset = load_split(args.dataset, args.split)
    dataset_length = len(dataset)
    if dataset_length < args.num_examples:
        raise ValueError(
            f"Requested {args.num_examples} examples but dataset split only has {dataset_length}.",
        )

    shuffled = dataset.shuffle(seed=args.seed)
    sampled = shuffled.select(range(args.num_examples))
    split_dict = sampled.train_test_split(
        test_size=args.eval_size,
        seed=args.seed,
        shuffle=False,  # already shuffled above for reproducibility
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / args.train_name
    eval_path = output_dir / args.eval_name

    split_dict["train"].save_to_disk(str(train_path))
    split_dict["test"].save_to_disk(str(eval_path))

    print(
        f"Saved {len(split_dict['train'])} training examples to {train_path} "
        f"and {len(split_dict['test'])} eval examples to {eval_path}.",
    )


if __name__ == "__main__":
    main()
