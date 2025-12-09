#!/usr/bin/env python3
"""Convert Sujet-Finance-Instruct tasks into ChatML JSONL files."""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

DEFAULT_TASKS = ["qa_with_context", "qa_conversation"]
DEFAULT_VAL_RATIO = 0.05
DEFAULT_SYSTEM_PROMPT = "You are a detail-oriented financial assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Sujet-Finance-Instruct by task type and export ChatML files.",
    )
    parser.add_argument(
        "--dataset",
        default="sujet-ai/Sujet-Finance-Instruct-177k",
        help="Hugging Face repo id or local path created via `save_to_disk`.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to convert (ignored for simple datasets saved locally).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task types to retain (match `task_type` column).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on total rows after filtering.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Fraction of examples reserved for validation (0 < ratio < 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic shuffling.",
    )
    parser.add_argument(
        "--output-dir",
        default="data_processed",
        help="Directory that will receive the ChatML jsonl files.",
    )
    parser.add_argument(
        "--system-prompt-fallback",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Used when a row has an empty `system_prompt`.",
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


def ensure_tasks_available(dataset: Dataset, requested: Sequence[str]) -> List[str]:
    present = set(dataset.unique("task_type"))
    missing = [task for task in requested if task not in present]
    if missing:
        raise ValueError(f"Tasks not present in dataset: {missing}")
    return list(requested)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("`--val-ratio` must be strictly between 0 and 1.")
    if not args.tasks:
        raise ValueError("Provide at least one task via `--tasks`.")

    dataset = load_split(args.dataset, args.split)
    tasks = ensure_tasks_available(dataset, args.tasks)
    task_set = set(tasks)
    dataset = dataset.filter(lambda row: row["task_type"] in task_set)

    if len(dataset) == 0:
        raise ValueError(f"No rows left after filtering tasks={tasks}.")

    dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))

    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    if n_val >= n_total:
        raise ValueError(
            "Validation split would consume the entire dataset. Reduce `--val-ratio` or supply more rows.",
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(tasks)
    train_path = out_dir / f"train_sujet_{suffix}_chatml.jsonl"
    val_path = out_dir / f"val_sujet_{suffix}_chatml.jsonl"

    def example_to_chatml(example):
        system_prompt = (example.get("system_prompt") or "").strip()
        user_prompt = (example.get("user_prompt") or "").strip()
        inputs = (example.get("inputs") or "").strip()
        answer = (example.get("answer") or "").strip()

        system_content = system_prompt or args.system_prompt_fallback
        user_content = user_prompt or inputs
        if not user_content:
            raise ValueError("Encountered row without user content.")

        return {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ],
        }

    split_idx = n_total - n_val
    with train_path.open("w", encoding="utf-8") as f_train, val_path.open(
        "w",
        encoding="utf-8",
    ) as f_val:
        for i, example in enumerate(dataset):
            chatml = example_to_chatml(example)
            line = json.dumps(chatml, ensure_ascii=False)
            if i < split_idx:
                f_train.write(line + "\n")
            else:
                f_val.write(line + "\n")

    print(
        f"Saved {split_idx} train and {n_val} val examples "
        f"for tasks={tasks} from {args.dataset}.",
    )
    print("Train:", train_path.resolve())
    print("Val:", val_path.resolve())


if __name__ == "__main__":
    main()
