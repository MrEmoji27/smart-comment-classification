"""
Fine-tune ModernBERT for SCC sentiment classification.

Expected labels: Positive, Neutral, Negative

Example:
    python backend/training/train_modernbert_sentiment.py ^
      --train-file test_comments.csv ^
      --text-column comment ^
      --label-column label_true

    python backend/training/train_modernbert_sentiment.py ^
      --dataset-name cardiffnlp/tweet_eval ^
      --dataset-config sentiment ^
      --text-column text ^
      --label-column label
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABEL_TO_ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
DEFAULT_BASE_MODEL = "answerdotai/ModernBERT-base"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models" / "modernbert-sentiment"
TWEET_EVAL_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ModernBERT for SCC sentiment classification.")
    parser.add_argument("--train-file", help="Path to CSV/XLSX/TXT training data.")
    parser.add_argument("--eval-file", help="Optional evaluation file. If omitted, a validation split is created from train data.")
    parser.add_argument("--dataset-name", help="Optional Hugging Face dataset name, e.g. cardiffnlp/tweet_eval.")
    parser.add_argument("--dataset-config", help="Optional Hugging Face dataset config, e.g. sentiment.")
    parser.add_argument("--train-split", default="train", help="Dataset split name for training.")
    parser.add_argument("--eval-split", help="Dataset split name for evaluation. If omitted, uses validation or test when available.")
    parser.add_argument("--text-column", default="comment", help="Column containing text.")
    parser.add_argument("--label-column", default="label_true", help="Column containing sentiment labels.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base ModernBERT checkpoint.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save the fine-tuned model.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Per-device train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Per-device eval batch size.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio when --eval-file is omitted.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    if not args.train_file and not args.dataset_name:
        parser.error("Provide either --train-file or --dataset-name.")
    return args


def load_table(path: str) -> pd.DataFrame:
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(src)
    if suffix == ".xlsx":
        return pd.read_excel(src, engine="openpyxl")
    if suffix == ".txt":
        return pd.DataFrame({"comment": [line.strip() for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]})
    raise ValueError(f"Unsupported data file: {src}")


def normalize_label(label: Any) -> str:
    cleaned = str(label).strip().capitalize()
    if cleaned not in LABEL_TO_ID:
        raise ValueError(f"Unsupported label '{label}'. Expected one of {sorted(LABEL_TO_ID)}")
    return cleaned


def compute_macro_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    predicted = predictions.argmax(axis=-1)
    f1_scores = []
    for label_id in sorted(ID_TO_LABEL):
        tp = int(((predicted == label_id) & (labels == label_id)).sum())
        fp = int(((predicted == label_id) & (labels != label_id)).sum())
        fn = int(((predicted != label_id) & (labels == label_id)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_scores) / len(f1_scores))


def build_examples(df: pd.DataFrame, text_column: str, label_column: str) -> list[dict[str, Any]]:
    if text_column not in df.columns:
        raise KeyError(f"Missing text column '{text_column}'")
    if label_column not in df.columns:
        raise KeyError(f"Missing label column '{label_column}'")

    examples = []
    for _, row in df[[text_column, label_column]].dropna().iterrows():
        text = str(row[text_column]).strip()
        if not text:
            continue
        label = normalize_label(row[label_column])
        examples.append({"text": text, "label": LABEL_TO_ID[label]})
    if not examples:
        raise ValueError("No valid training examples found after cleaning.")
    return examples


def build_examples_from_hf(records, text_column: str, label_column: str) -> list[dict[str, Any]]:
    examples = []
    for row in records:
        text = str(row[text_column]).strip()
        if not text:
            continue
        raw_label = row[label_column]
        if isinstance(raw_label, (int, np.integer)) and int(raw_label) in TWEET_EVAL_LABELS:
            label = TWEET_EVAL_LABELS[int(raw_label)]
        else:
            label = normalize_label(raw_label)
        examples.append({"text": text, "label": LABEL_TO_ID[label]})
    if not examples:
        raise ValueError("No valid Hugging Face dataset examples found after cleaning.")
    return examples


@dataclass
class EncodedSentimentDataset(Dataset):
    encodings: dict[str, Any]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def make_dataset(tokenizer, examples: list[dict[str, Any]], max_length: int) -> EncodedSentimentDataset:
    encodings = tokenizer(
        [example["text"] for example in examples],
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    labels = [example["label"] for example in examples]
    return EncodedSentimentDataset(encodings=encodings, labels=labels)


def split_examples(examples: list[dict[str, Any]], validation_split: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    shuffled_indices = np.arange(len(examples))
    rng.shuffle(shuffled_indices)
    split_index = max(1, int(len(examples) * (1 - validation_split)))
    train_idx = shuffled_indices[:split_index]
    eval_idx = shuffled_indices[split_index:]
    if len(eval_idx) == 0:
        eval_idx = shuffled_indices[-1:]
        train_idx = shuffled_indices[:-1]
    train_examples = [examples[i] for i in train_idx]
    eval_examples = [examples[i] for i in eval_idx]
    return train_examples, eval_examples


def load_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if args.dataset_name:
        dataset = load_dataset(args.dataset_name, args.dataset_config)
        if args.train_split not in dataset:
            raise KeyError(f"Missing train split '{args.train_split}' in dataset {args.dataset_name}")

        train_examples = build_examples_from_hf(dataset[args.train_split], args.text_column, args.label_column)

        if args.eval_split:
            if args.eval_split not in dataset:
                raise KeyError(f"Missing eval split '{args.eval_split}' in dataset {args.dataset_name}")
            eval_examples = build_examples_from_hf(dataset[args.eval_split], args.text_column, args.label_column)
        elif "validation" in dataset:
            eval_examples = build_examples_from_hf(dataset["validation"], args.text_column, args.label_column)
        elif "test" in dataset:
            eval_examples = build_examples_from_hf(dataset["test"], args.text_column, args.label_column)
        else:
            train_examples, eval_examples = split_examples(train_examples, args.validation_split, args.seed)
        return train_examples, eval_examples

    train_df = load_table(args.train_file)
    train_examples = build_examples(train_df, args.text_column, args.label_column)

    if args.eval_file:
        eval_df = load_table(args.eval_file)
        eval_examples = build_examples(eval_df, args.text_column, args.label_column)
    else:
        train_examples, eval_examples = split_examples(train_examples, args.validation_split, args.seed)
    return train_examples, eval_examples


def main() -> None:
    args = parse_args()
    train_examples, eval_examples = load_examples(args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    train_dataset = make_dataset(tokenizer, train_examples, args.max_length)
    eval_dataset = make_dataset(tokenizer, eval_examples, args.max_length)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        seed=args.seed,
        report_to="none",
        save_total_limit=2,
        do_train=True,
        do_eval=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.asarray(logits)
        labels = np.asarray(labels)
        accuracy = float((predictions.argmax(axis=-1) == labels).mean())
        macro_f1 = compute_macro_f1(predictions, labels)
        return {"accuracy": accuracy, "macro_f1": macro_f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "base_model": args.base_model,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "labels": ID_TO_LABEL,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "metrics": metrics,
    }
    (output_dir / "scc_model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved ModernBERT checkpoint to: {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
