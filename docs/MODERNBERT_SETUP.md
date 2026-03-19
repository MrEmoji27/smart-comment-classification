# ModernBERT Setup

This repo now supports ModernBERT as the preferred sentiment backend, but you still need a fine-tuned checkpoint.

## 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

## 2. Prepare Training Data

Use a CSV or XLSX file with at least:

- `comment`
- `label_true`

Supported sentiment labels:

- `Positive`
- `Neutral`
- `Negative`

The sample file [test_comments.csv](/c:/Users/mremo/OneDrive/Desktop/scc/scc/test_comments.csv) is only a tiny smoke-test dataset, not enough for production training.

## 3. Train ModernBERT

### Option A: Train From A Public Dataset

The training script can now pull `cardiffnlp/tweet_eval` directly from Hugging Face:

```bash
python backend/training/train_modernbert_sentiment.py ^
  --dataset-name cardiffnlp/tweet_eval ^
  --dataset-config sentiment ^
  --text-column text ^
  --label-column label ^
  --output-dir backend/models/modernbert-sentiment
```

This is the fastest way to create a real ModernBERT checkpoint for the project.

### Option B: Train From Local Data

```bash
python backend/training/train_modernbert_sentiment.py ^
  --train-file test_comments.csv ^
  --text-column comment ^
  --label-column label_true ^
  --output-dir backend/models/modernbert-sentiment
```

Optional:

- Pass `--eval-file` if you have a separate validation split.
- Pass `--base-model` to use a different ModernBERT checkpoint.
- Pass `--eval-split` if your Hugging Face dataset uses a custom validation split name.

## 4. Activate ModernBERT In The App

You have two options.

### Option A: Local checkpoint convention

If you train into:

```bash
backend/models/modernbert-sentiment
```

the backend will detect it automatically on startup.

### Option B: Explicit environment variable

```bash
set MODERNBERT_SENTIMENT_MODEL=backend/models/modernbert-sentiment
```

or

```bash
set MODERNBERT_SENTIMENT_MODEL=your-org/modernbert-comment-sentiment
```

## 5. Verify It Is Active

Start the backend and check:

```bash
curl http://localhost:8000/health
```

Look for:

- `preferred_sentiment_model`
- `active_sentiment_model`
- `active_sentiment_display_name`

If ModernBERT fails to load, the backend falls back to Twitter RoBERTa for sentiment and reports that honestly in `/health` and the frontend runtime label.

## What Is Still Needed For Production

- A real labeled sentiment dataset larger than the sample CSV
- Evaluation on held-out data
- Threshold tuning against your real comment domain
- Optional comparison against the current RoBERTa sentiment backend before switching permanently
