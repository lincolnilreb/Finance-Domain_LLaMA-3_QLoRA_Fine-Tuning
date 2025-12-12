# Finance-Domain LLaMA-3 QLoRA Fine-Tuning

Project scaffold for experimenting with QLoRA-based finance models.

## Structure
- `data_raw/`: unprocessed source datasets.
- `data_processed/`: cleaned/feature-engineered data ready for training.
- `scripts/`: utilities for preprocessing, training, or evaluation.
- `requirements.txt`: Python dependencies for the environment.

## Step 3 – Sampling Strategy
For a first training run we will:
- draw `N = 68,912` examples from the `train` split (this is the full dataset size),
- reserve `1,000` of those for evaluation, and
- use a reproducible random shuffle (default seed `42`).

### Download source data
Run:

```bash
python scripts/inspect_finance_alpaca.py --output-dir data_raw/finance_alpaca
```

This grabs `gbharti/finance-alpaca`, prints a preview, and persistently stores the dataset at `data_raw/finance_alpaca`, giving you a concrete path for downstream sampling.

Use `scripts/sample_split.py` to materialize those subsets. Example:

```bash
python scripts/sample_split.py \
  --dataset path_or_hf_repo_for_your_train_split \
  --split train \
  --num-examples 68912 \
  --eval-size 1000 \
  --seed 42 \
  --output-dir data_processed/sampling_v1
```

The script first shuffles the source split, samples the requested number of rows, then writes the resulting train/eval splits to `data_processed/sampling_v1/train_sample` and `data_processed/sampling_v1/eval_sample`. Both folders are Hugging Face `Dataset` objects saved via `save_to_disk`, so you can load them later with `datasets.load_from_disk`.

## Step 4 – Finance-Alpaca → ChatML
Produce ChatML-ready files for QLoRA training:

```bash
python scripts/make_chatml_finance_alpaca.py \
  --dataset data_raw/finance_alpaca \
  --split train \
  --max-samples 10000 \
  --val-ratio 0.05 \
  --output-dir data_processed
```

This shuffles the split, samples the requested rows, and writes `data_processed/train_finance_alpaca_chatml.jsonl` and `data_processed/val_finance_alpaca_chatml.jsonl`. Each line is a ChatML message trio (`system`/`user`/`assistant`) ready for LLaMA‑3 QLoRA.

## Step 5 – Multi-task augmentation

### 5.1 Sujet-Finance-Instruct-177k (Hugging Face)
- **Schema**: 177,597 instructions aggregated from 18 finance datasets spanning seven task families. Each row includes `system_prompt`, `user_prompt`, `answer`, source `dataset`, and a `task_type` label.
- **Tasks available**:
  1. `sentiment_analysis`
  2. `ner_sentiment_analysis`
  3. `qa`
  4. `qa_with_context`
  5. `qa_conversation`
  6. `topic_classification`
  7. `yes_no_question`

Pick one or two task types (e.g., QA vs. classification) to mirror the ChatML schema from Step 4. Next up we’ll add `scripts/make_chatml_sujet.py` to filter by the chosen `task_type`, normalize prompts into ChatML, and emit train/val JSONL files that can later be concatenated with the Finance-Alpaca outputs for a multi-task QLoRA fine-tune.

For QA-heavy coverage we target:

- `qa_with_context`
- `qa_conversation`

Convert them into ChatML by running:

```bash
python scripts/make_chatml_sujet.py \
  --dataset sujet-ai/Sujet-Finance-Instruct-177k \
  --tasks qa_with_context qa_conversation \
  --max-samples 20000 \
  --val-ratio 0.05 \
  --output-dir data_processed
```

The script filters by `task_type`, shuffles, optionally subsamples, then writes `train_sujet_qa_with_context_qa_conversation_chatml.jsonl` and `val_sujet_qa_with_context_qa_conversation_chatml.jsonl`. Later you can concatenate these JSONL files with the Finance-Alpaca ChatML files (simple file append) before launching QLoRA.

### 5.2 Merge ChatML corpora
Combine any number of JSONL shards with `scripts/merge_chatml.py`:

```bash
# Merge and shuffle training shards
python scripts/merge_chatml.py \
  --inputs data_processed/train_finance_alpaca_chatml.jsonl \
           data_processed/train_sujet_qa_with_context_qa_conversation_chatml.jsonl \
  --output data_processed/train_chatml_all.jsonl \
  --shuffle --seed 42

# Merge validation shards (keep order deterministic)
python scripts/merge_chatml.py \
  --inputs data_processed/val_finance_alpaca_chatml.jsonl \
           data_processed/val_sujet_qa_with_context_qa_conversation_chatml.jsonl \
  --output data_processed/val_chatml_all.jsonl
```

The script loads every line from the inputs, optionally shuffles them, and streams to the final JSONL. Adjust file paths or add more `--inputs` as your corpus grows.
