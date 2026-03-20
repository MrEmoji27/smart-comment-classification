"""
Fine-tune ModernBERT for SCC sentiment classification.

Expected labels: Positive, Neutral, Negative

Single-dataset examples:
    python backend/training/train_modernbert_sentiment.py ^
      --train-file test_comments.csv ^
      --text-column comment ^
      --label-column label_true

    python backend/training/train_modernbert_sentiment.py ^
      --dataset-name cardiffnlp/tweet_eval ^
      --dataset-config sentiment ^
      --text-column text ^
      --label-column label

Multi-dataset preset examples:
    python backend/training/train_modernbert_sentiment.py --preset mixed
    python backend/training/train_modernbert_sentiment.py --preset wide --max-samples-per-dataset 8000
    python backend/training/train_modernbert_sentiment.py --preset reviews
    python backend/training/train_modernbert_sentiment.py --preset financial
    python backend/training/train_modernbert_sentiment.py --preset academic
    python backend/training/train_modernbert_sentiment.py --preset social

Custom multi-dataset example:
    python backend/training/train_modernbert_sentiment.py ^
      --multi-dataset cardiffnlp/tweet_eval:sentiment:text:label:3class ^
      --multi-dataset imdb::text:label:binary ^
      --multi-dataset rotten_tomatoes::text:label:binary ^
      --max-samples-per-dataset 20000 ^
      --class-weights

Label schema values for --multi-dataset:
  binary  : 0=Negative, 1=Positive  (no Neutral; e.g. imdb, yelp_polarity)
  3class  : 0=Negative, 1=Neutral, 2=Positive  (tweet_eval style)
  5class  : 0,1=Negative, 2=Neutral, 3,4=Positive  (SST-5 style)
  stars   : 1,2=Negative, 3=Neutral, 4,5=Positive  (star-rating style)
  auto    : tries 3class mapping, falls back to string normalization
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

LABEL_TO_ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
DEFAULT_BASE_MODEL = "answerdotai/ModernBERT-base"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "models" / "modernbert-sentiment"

# ── Label schema maps ──────────────────────────────────────────────────────────
# tweet_eval / 3-class: 0=Negative, 1=Neutral, 2=Positive
THREE_CLASS_LABEL_MAP: dict[int, str] = {0: "Negative", 1: "Neutral", 2: "Positive"}
# Legacy alias kept for backward compatibility
TWEET_EVAL_LABELS = THREE_CLASS_LABEL_MAP
# Binary datasets (imdb, yelp_polarity, rotten_tomatoes): 0=Negative, 1=Positive
BINARY_LABEL_MAP: dict[int, str] = {0: "Negative", 1: "Positive"}
# SST-5 style: 0,1=Negative, 2=Neutral, 3,4=Positive
FIVE_CLASS_LABEL_MAP: dict[int, str] = {0: "Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Positive"}
# Star-rating (1-5): 1,2=Negative, 3=Neutral, 4,5=Positive
STAR_LABEL_MAP: dict[int, str] = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}

_SCHEMA_MAPS: dict[str, dict[int, str]] = {
    "binary": BINARY_LABEL_MAP,
    "3class": THREE_CLASS_LABEL_MAP,
    "5class": FIVE_CLASS_LABEL_MAP,
    "stars": STAR_LABEL_MAP,
}

# String label aliases (lowercase key → canonical label)
STRING_LABEL_ALIASES: dict[str, str] = {
    "positive": "Positive", "pos": "Positive",
    "negative": "Negative", "neg": "Negative",
    "neutral": "Neutral", "neu": "Neutral",
    "very positive": "Positive", "strongly positive": "Positive",
    "somewhat positive": "Positive", "slightly positive": "Positive",
    "very negative": "Negative", "strongly negative": "Negative",
    "somewhat negative": "Negative", "slightly negative": "Negative",
    "mixed": "Neutral", "ambiguous": "Neutral", "objective": "Neutral",
}

# ── Curated dataset presets ────────────────────────────────────────────────────
# Each entry: (hf_name, hf_config_or_None, text_col, label_col, schema)
#
# Schema reference:
#   binary  0=Negative, 1=Positive
#   3class  0=Negative, 1=Neutral,  2=Positive   (tweet_eval / financial_phrasebank)
#   5class  0,1=Negative, 2=Neutral, 3,4=Positive (SST-5 / yelp_review_full 0-indexed)
#   stars   1,2=Negative, 3=Neutral, 4,5=Positive (app_reviews raw star value)
#   auto    tries 3class map first, then string aliases
DATASET_PRESETS: dict[str, list[tuple[str, Optional[str], str, str, str]]] = {

    # ── Social media ──────────────────────────────────────────────────────────
    # Covers tweets, casual language, slang, abbreviations
    "social": [
        ("cardiffnlp/tweet_eval",             "sentiment", "text",     "label", "3class"),
        ("mteb/tweet_sentiment_multilingual",  "english",   "text",     "label", "3class"),
    ],

    # ── Review corpora ────────────────────────────────────────────────────────
    # Movies, restaurants, products, mobile apps — consumer opinion language
    "reviews": [
        ("rotten_tomatoes",  None, "text",    "label", "binary"),  # movie critics
        ("imdb",             None, "text",    "label", "binary"),  # movie audience
        ("yelp_polarity",    None, "text",    "label", "binary"),  # restaurant/business
        ("amazon_polarity",  None, "content", "label", "binary"),  # product reviews (3.6M — cap with --max-samples-per-dataset)
        ("yelp_review_full", None, "text",    "label", "5class"),  # full 1-5 star Yelp (0-indexed)
        ("app_reviews",      None, "review",  "star",  "stars"),   # Google Play app reviews
    ],

    # ── Academic / literary ───────────────────────────────────────────────────
    # Stanford Sentiment Treebank — fine-grained sentence-level annotations
    "academic": [
        ("glue",        "sst2", "sentence", "label", "binary"),  # SST binary (~67k)
        ("SetFit/sst5", None,   "text",     "label", "5class"),  # SST-5 fine-grained
        ("rotten_tomatoes", None, "text",   "label", "binary"),  # critic snippets
    ],

    # ── Financial / news ─────────────────────────────────────────────────────
    # Formal language, finance tweets, news headlines
    "financial": [
        ("zeroshot/twitter-financial-news-sentiment", None, "text", "label", "3class"),
        ("cardiffnlp/tweet_eval",                     "sentiment", "text", "label", "3class"),
    ],

    # ── Mixed (cross-domain, balanced) ───────────────────────────────────────
    # Good general-purpose model. Use --max-samples-per-dataset 10000.
    "mixed": [
        ("cardiffnlp/tweet_eval",                     "sentiment", "text",    "label",  "3class"),
        ("mteb/tweet_sentiment_multilingual",          "english",   "text",    "label",  "3class"),
        ("rotten_tomatoes",                           None,        "text",    "label",  "binary"),
        ("imdb",                                      None,        "text",    "label",  "binary"),
        ("yelp_polarity",                             None,        "text",    "label",  "binary"),
        ("amazon_polarity",                           None,        "content", "label",  "binary"),
        ("zeroshot/twitter-financial-news-sentiment", None,        "text",    "label",  "3class"),
        ("app_reviews",                               None,        "review",  "star",   "stars"),
        ("SetFit/sst5",                               None,        "text",    "label",  "5class"),
    ],

    # ── Wide (maximum domain coverage) ───────────────────────────────────────
    # Everything. Use --max-samples-per-dataset 3000 for ~3-4 hour training.
    "wide": [
        # Social media
        ("cardiffnlp/tweet_eval",                     "sentiment", "text",    "label",  "3class"),
        ("mteb/tweet_sentiment_multilingual",          "english",   "text",    "label",  "3class"),
        # Movies / entertainment
        ("imdb",                                      None,        "text",    "label",  "binary"),
        ("rotten_tomatoes",                           None,        "text",    "label",  "binary"),
        ("glue",                                      "sst2",      "sentence","label",  "binary"),
        ("SetFit/sst5",                               None,        "text",    "label",  "5class"),
        # Restaurants / local business
        ("yelp_polarity",                             None,        "text",    "label",  "binary"),
        ("yelp_review_full",                          None,        "text",    "label",  "5class"),
        # E-commerce / products
        ("amazon_polarity",                           None,        "content", "label",  "binary"),
        # Mobile apps / tech
        ("app_reviews",                               None,        "review",  "star",   "stars"),
        # Finance / news (parquet-native, no legacy scripts)
        ("zeroshot/twitter-financial-news-sentiment", None,        "text",    "label",  "3class"),
    ],

    # ── Slang (internet/informal language focus) ──────────────────────────────
    # Good for classifying casual text, Gen-Z language, meme-speak, slang.
    "slang": [
        ("cardiffnlp/tweet_eval",            "sentiment", "text",    "label", "3class"),
        ("mteb/tweet_sentiment_multilingual", "english",  "text",    "label", "3class"),
    ],
}


# ── Universal label normalizer ─────────────────────────────────────────────────

def normalize_label_universal(label: Any, schema: str = "auto") -> Optional[str]:
    """
    Normalize a raw dataset label to 'Positive', 'Neutral', or 'Negative'.
    Returns None if the label cannot be mapped (row is silently dropped).

    schema options: auto | binary | 3class | 5class | stars
    """
    if isinstance(label, str):
        cleaned = label.strip().lower()
        if cleaned in STRING_LABEL_ALIASES:
            return STRING_LABEL_ALIASES[cleaned]
        capitalized = label.strip().capitalize()
        if capitalized in LABEL_TO_ID:
            return capitalized
        LOGGER.debug("Unknown string label: %r", label)
        return None

    try:
        int_label = int(label)
    except (TypeError, ValueError):
        LOGGER.debug("Cannot convert label to int: %r", label)
        return None

    if schema == "auto":
        # Default to 3class; handles tweet_eval, financial_phrasebank, etc.
        return THREE_CLASS_LABEL_MAP.get(int_label)

    mapping = _SCHEMA_MAPS.get(schema)
    if mapping is None:
        raise ValueError(f"Unknown label schema: {schema!r}. Choose from: {sorted(_SCHEMA_MAPS)}")
    return mapping.get(int_label)


# Backward-compatible wrapper used by build_examples_from_hf (original signature)
def normalize_label(label: Any) -> str:
    cleaned = str(label).strip().capitalize()
    if cleaned not in LABEL_TO_ID:
        raise ValueError(f"Unsupported label '{label}'. Expected one of {sorted(LABEL_TO_ID)}")
    return cleaned


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ModernBERT for SCC sentiment classification.")

    # ── Single-source inputs (original flags) ──
    single = parser.add_argument_group("Single-source inputs (original)")
    single.add_argument("--train-file", help="Path to CSV/XLSX/TXT training data.")
    single.add_argument("--eval-file", help="Optional evaluation file.")
    single.add_argument("--dataset-name", help="HuggingFace dataset name, e.g. cardiffnlp/tweet_eval.")
    single.add_argument("--dataset-config", help="HuggingFace dataset config, e.g. sentiment.")
    single.add_argument("--train-split", default="train", help="Dataset split name for training.")
    single.add_argument("--eval-split", help="Dataset split name for evaluation.")
    single.add_argument("--text-column", default="comment", help="Column containing text.")
    single.add_argument("--label-column", default="label_true", help="Column containing sentiment labels.")

    # ── Multi-dataset inputs (new) ──
    multi = parser.add_argument_group("Multi-dataset inputs (new)")
    multi.add_argument(
        "--preset",
        choices=list(DATASET_PRESETS.keys()),
        help=(
            "Use a predefined multi-dataset configuration. "
            f"Available: {', '.join(DATASET_PRESETS)}."
        ),
    )
    multi.add_argument(
        "--multi-dataset",
        action="append",
        default=[],
        metavar="NAME:CONFIG:TEXT_COL:LABEL_COL:SCHEMA",
        help=(
            "Add a HuggingFace dataset to the training mix. "
            "Format: name:config:text_col:label_col:schema "
            "(use empty string for config if none). "
            "Can be repeated. "
            "schema = auto|binary|3class|5class|stars."
        ),
    )
    multi.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Cap examples taken from each dataset. Prevents large datasets from dominating.",
    )
    multi.add_argument(
        "--class-weights",
        action="store_true",
        help="Use class-weighted cross-entropy loss to handle imbalanced label distributions.",
    )

    # ── Model and training hyperparameters ──
    hp = parser.add_argument_group("Model / training hyperparameters")
    hp.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base ModernBERT checkpoint.")
    hp.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save the fine-tuned model.")
    hp.add_argument("--max-length", type=int, default=512, help="Maximum token length.")
    hp.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    hp.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    hp.add_argument("--train-batch-size", type=int, default=8, help="Per-device train batch size.")
    hp.add_argument("--eval-batch-size", type=int, default=8, help="Per-device eval batch size.")
    hp.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    hp.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation steps.")
    hp.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio.")
    hp.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    has_input = bool(
        args.train_file or args.dataset_name or args.preset or args.multi_dataset
    )
    if not has_input:
        parser.error(
            "Provide one of: --train-file, --dataset-name, --preset, or --multi-dataset."
        )
    return args


# ── File / table loading ───────────────────────────────────────────────────────

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


# ── Example builders ───────────────────────────────────────────────────────────

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
    """Original HF builder — handles tweet_eval-style integer labels (0/1/2) and string labels."""
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
        raise ValueError("No valid HuggingFace dataset examples found after cleaning.")
    return examples


def build_examples_from_hf_schema(
    records,
    text_column: str,
    label_column: str,
    schema: str = "auto",
    max_samples: Optional[int] = None,
    dataset_tag: str = "",
) -> list[dict[str, Any]]:
    """
    HF builder with universal label normalization.

    Supports binary, 3class, 5class, star-rating, and string labels.
    Rows with unmappable labels are silently dropped.
    """
    examples = []
    skipped = 0
    for row in records:
        text = str(row[text_column]).strip()
        if not text:
            continue
        raw_label = row[label_column]
        label = normalize_label_universal(raw_label, schema)
        if label is None:
            skipped += 1
            continue
        examples.append({"text": text, "label": LABEL_TO_ID[label]})
        if max_samples and len(examples) >= max_samples:
            break

    if skipped:
        LOGGER.warning("[%s] Dropped %d rows with unmappable labels (schema=%s).", dataset_tag or "dataset", skipped, schema)
    if not examples:
        raise ValueError(f"No valid examples from dataset '{dataset_tag}' after label normalization (schema={schema}).")

    LOGGER.info("[%s] Loaded %d examples (schema=%s).", dataset_tag or "dataset", len(examples), schema)
    return examples


# ── Example splitting ──────────────────────────────────────────────────────────

def split_examples(
    examples: list[dict[str, Any]],
    validation_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    shuffled_indices = np.arange(len(examples))
    rng.shuffle(shuffled_indices)
    split_index = max(1, int(len(examples) * (1 - validation_split)))
    train_idx = shuffled_indices[:split_index]
    eval_idx = shuffled_indices[split_index:]
    if len(eval_idx) == 0:
        eval_idx = shuffled_indices[-1:]
        train_idx = shuffled_indices[:-1]
    return [examples[i] for i in train_idx], [examples[i] for i in eval_idx]


# ── Multi-dataset loaders ──────────────────────────────────────────────────────

def _load_one_hf_dataset(
    hf_name: str,
    hf_config: Optional[str],
    text_col: str,
    label_col: str,
    schema: str,
    train_split: str,
    max_samples: Optional[int],
    validation_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tag = f"{hf_name}/{hf_config}" if hf_config else hf_name
    LOGGER.info("Loading HF dataset: %s (schema=%s)", tag, schema)
    dataset = load_dataset(hf_name, hf_config)

    train_examples = build_examples_from_hf_schema(
        dataset[train_split],
        text_col,
        label_col,
        schema=schema,
        max_samples=max_samples,
        dataset_tag=tag,
    )

    # Prefer a built-in validation/test split; fall back to manual split
    if "validation" in dataset:
        eval_records = dataset["validation"]
    elif "test" in dataset:
        eval_records = dataset["test"]
    else:
        train_examples, eval_examples = split_examples(train_examples, validation_split, seed)
        return train_examples, eval_examples

    eval_examples = build_examples_from_hf_schema(
        eval_records,
        text_col,
        label_col,
        schema=schema,
        max_samples=max_samples,
        dataset_tag=f"{tag}/eval",
    )
    return train_examples, eval_examples


# Urban Dictionary constants
URBAN_DICT_HF_NAME = "rexarski/urban-dictionary-words"
URBAN_DICT_MIN_VOTES = 50  # ignore entries with fewer combined votes (noisy)


def load_urban_dict_examples(
    max_samples: Optional[int],
    validation_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Load Urban Dictionary and derive sentiment from community voting ratio.

    Label derivation (thumbs_up / (thumbs_up + thumbs_down)):
      >= 0.70  → Positive   (crowd-approved slang / positive meaning)
      <= 0.30  → Negative   (crowd-disapproved / offensive / negative meaning)
      else     → Neutral    (contested or ambiguous)

    Only entries with at least URBAN_DICT_MIN_VOTES total votes are kept
    to filter out low-quality / spam definitions.
    """
    LOGGER.info("Loading Urban Dictionary dataset (schema=thumbs)")
    dataset = load_dataset(URBAN_DICT_HF_NAME)

    # Urban Dictionary has a single "train" split
    split_key = "train" if "train" in dataset else list(dataset.keys())[0]
    records = dataset[split_key]

    examples: list[dict[str, Any]] = []
    for row in records:
        text = (row.get("example") or "").strip()
        if not text:
            continue

        up = int(row.get("thumbs_up") or 0)
        down = int(row.get("thumbs_down") or 0)
        total = up + down
        if total < URBAN_DICT_MIN_VOTES:
            continue

        ratio = up / total
        if ratio >= 0.70:
            sentiment = "Positive"
        elif ratio <= 0.30:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        examples.append({
            "text": text,
            "label": LABEL_TO_ID[sentiment],
            "label_str": sentiment,
            "_source": URBAN_DICT_HF_NAME,
        })

    LOGGER.info("Urban Dictionary: %d usable examples (min_votes=%d).", len(examples), URBAN_DICT_MIN_VOTES)

    if max_samples and len(examples) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(examples), size=max_samples, replace=False)
        examples = [examples[i] for i in idx]

    return split_examples(examples, validation_split, seed)


def _load_dataset_entry(
    hf_name: str,
    hf_config: Optional[str],
    text_col: str,
    label_col: str,
    schema: str,
    train_split: str,
    max_samples: Optional[int],
    validation_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Route to specialised loader when schema == 'thumbs', else generic HF loader."""
    if schema == "thumbs":
        return load_urban_dict_examples(max_samples, validation_split, seed)
    return _load_one_hf_dataset(
        hf_name, hf_config, text_col, label_col, schema,
        train_split, max_samples, validation_split, seed,
    )


def load_preset_examples(
    preset_name: str,
    train_split: str,
    validation_split: float,
    seed: int,
    max_samples: Optional[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = DATASET_PRESETS[preset_name]
    LOGGER.info("Using preset '%s' with %d dataset(s).", preset_name, len(entries))
    all_train: list[dict[str, Any]] = []
    all_eval: list[dict[str, Any]] = []

    for hf_name, hf_config, text_col, label_col, schema in entries:
        try:
            train_ex, eval_ex = _load_dataset_entry(
                hf_name, hf_config, text_col, label_col, schema,
                train_split, max_samples, validation_split, seed,
            )
            all_train.extend(train_ex)
            all_eval.extend(eval_ex)
        except Exception as exc:
            LOGGER.error("Failed to load dataset '%s/%s': %s. Skipping.", hf_name, hf_config, exc)

    if not all_train:
        raise RuntimeError(f"No training examples loaded for preset '{preset_name}'.")
    LOGGER.info("Preset '%s': %d train, %d eval examples total.", preset_name, len(all_train), len(all_eval))
    return all_train, all_eval


def _parse_multi_dataset_spec(spec: str) -> tuple[str, Optional[str], str, str, str]:
    """Parse 'name:config:text_col:label_col:schema' — config may be empty."""
    parts = spec.split(":")
    if len(parts) != 5:
        raise ValueError(
            f"--multi-dataset spec must have exactly 5 colon-separated parts: "
            f"name:config:text_col:label_col:schema. Got: {spec!r}"
        )
    hf_name, hf_config, text_col, label_col, schema = parts
    return hf_name, (hf_config or None), text_col, label_col, (schema or "auto")


def load_multi_examples(
    specs: list[str],
    train_split: str,
    validation_split: float,
    seed: int,
    max_samples: Optional[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_train: list[dict[str, Any]] = []
    all_eval: list[dict[str, Any]] = []

    for spec in specs:
        hf_name, hf_config, text_col, label_col, schema = _parse_multi_dataset_spec(spec)
        try:
            train_ex, eval_ex = _load_dataset_entry(
                hf_name, hf_config, text_col, label_col, schema,
                train_split, max_samples, validation_split, seed,
            )
            all_train.extend(train_ex)
            all_eval.extend(eval_ex)
        except Exception as exc:
            LOGGER.error("Failed to load dataset from spec '%s': %s. Skipping.", spec, exc)

    if not all_train:
        raise RuntimeError("No training examples loaded from --multi-dataset specs.")
    LOGGER.info("Multi-dataset: %d train, %d eval examples total.", len(all_train), len(all_eval))
    return all_train, all_eval


# ── Main example loading router ────────────────────────────────────────────────

def load_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Preset takes priority if both preset and multi-dataset are given
    if args.preset:
        return load_preset_examples(
            args.preset,
            train_split=args.train_split,
            validation_split=args.validation_split,
            seed=args.seed,
            max_samples=args.max_samples_per_dataset,
        )

    if args.multi_dataset:
        return load_multi_examples(
            args.multi_dataset,
            train_split=args.train_split,
            validation_split=args.validation_split,
            seed=args.seed,
            max_samples=args.max_samples_per_dataset,
        )

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

    # Local file fallback
    train_df = load_table(args.train_file)
    train_examples = build_examples(train_df, args.text_column, args.label_column)

    if args.eval_file:
        eval_df = load_table(args.eval_file)
        eval_examples = build_examples(eval_df, args.text_column, args.label_column)
    else:
        train_examples, eval_examples = split_examples(train_examples, args.validation_split, args.seed)
    return train_examples, eval_examples


# ── Dataset + metrics ──────────────────────────────────────────────────────────

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


# ── Weighted-loss trainer ──────────────────────────────────────────────────────

class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies class weights to cross-entropy loss."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # Copy to avoid mutating the shared inputs dict with .pop()
        inputs = {k: v for k, v in inputs.items()}
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(examples: list[dict[str, Any]]) -> torch.Tensor:
    """Inverse-frequency class weights for balanced training."""
    counts = Counter(ex["label"] for ex in examples)
    total = sum(counts.values())
    weights = [total / (len(LABEL_TO_ID) * counts.get(i, 1)) for i in range(len(LABEL_TO_ID))]
    LOGGER.info(
        "Class distribution — %s",
        {ID_TO_LABEL[i]: counts.get(i, 0) for i in range(len(LABEL_TO_ID))},
    )
    LOGGER.info(
        "Class weights — %s",
        {ID_TO_LABEL[i]: round(w, 4) for i, w in enumerate(weights)},
    )
    return torch.tensor(weights, dtype=torch.float32)


# ── Main ───────────────────────────────────────────────────────────────────────

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

    steps_per_epoch = max(1, len(train_dataset) // args.train_batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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

    class_weights = compute_class_weights(train_examples) if args.class_weights else None
    TrainerClass = WeightedLossTrainer if args.class_weights else Trainer

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        **({"class_weights": class_weights} if args.class_weights else {}),
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Determine data source description for metadata
    if args.preset:
        data_source = {"type": "preset", "preset": args.preset}
    elif args.multi_dataset:
        data_source = {"type": "multi_dataset", "specs": args.multi_dataset}
    elif args.dataset_name:
        data_source = {"type": "hf_dataset", "name": args.dataset_name, "config": args.dataset_config}
    else:
        data_source = {"type": "local_file", "path": args.train_file}

    metadata = {
        "base_model": args.base_model,
        "data_source": data_source,
        "labels": ID_TO_LABEL,
        "class_weights_used": args.class_weights,
        "max_samples_per_dataset": args.max_samples_per_dataset,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "label_distribution": {
            "train": {ID_TO_LABEL[k]: v for k, v in Counter(ex["label"] for ex in train_examples).items()},
            "eval": {ID_TO_LABEL[k]: v for k, v in Counter(ex["label"] for ex in eval_examples).items()},
        },
        "metrics": metrics,
    }
    (output_dir / "scc_model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved ModernBERT checkpoint to: {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
