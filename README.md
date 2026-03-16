# Smart Comment Classification

> AI-powered comment analysis that understands real-world English — slang, emojis, sarcasm, typos, and all.

Classify any comment by **sentiment**, **type**, **toxicity**, **emotion**, and **sarcasm** in real time. Built with FastAPI + 5 HuggingFace models on the backend, React on the frontend.

---

## Features

- **Sentiment analysis** — Positive, Neutral, or Negative with confidence scores
- **Comment type** — Praise, Complaint, Question, Feedback, Spam, or Other
- **Toxicity detection** — scores abusive/harmful language 0–100%
- **Emotion detection** — identifies up to 28 fine-grained emotions (joy, anger, curiosity, gratitude…)
- **Sarcasm detection** — catches ironic positivity and flips sentiment accordingly
- **Word-level highlighting** — every word color-coded by its individual sentiment
- **Multi-sentence breakdown** — detects mixed sentiment across sentences
- **Batch file classification** — upload CSV, TXT, or XLSX files for bulk analysis
- **Gibberish detection** — rejects keyboard mashing and numeric spam before hitting any model
- **Informal English support** — handles slang (`ngl`, `fire`, `bussin`), contractions, abbreviations, emojis

---

## Demo

### Single comment
Type or paste any comment and get instant analysis:

```
Input:  "Oh wow this product is SO amazing, it only crashed 5 times today!"

Output:
  Sentiment   → Negative  (sarcasm detected — 87%)
  Type        → Complaint
  Emotions    → admiration 90%, surprise 9%
  Toxicity    → 0.07%
```

### Batch file
Upload a `.csv` with a column of comments — get a downloadable results table with charts.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Backend API | Python · FastAPI · Uvicorn |
| ML | HuggingFace Transformers · PyTorch (CPU) |
| Frontend | React 19 · Vite |
| HTTP | Axios |
| Lexicon | VADER Sentiment |

---

## Models

| Model | Task | Parameters |
|---|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment | 125M |
| `unitary/toxic-bert` | Toxicity | 110M |
| `facebook/bart-large-mnli` | Comment type (zero-shot) | 400M |
| `SamLowe/roberta-base-go_emotions` | Emotion (28 classes) | 125M |
| `cardiffnlp/twitter-roberta-base-irony` | Sarcasm/irony | 125M |
| VADER lexicon | Word-level sentiment | rule-based |

> First run downloads ~1.5 GB of model weights, cached locally after that.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Server starts at `http://localhost:8000`. The first startup takes 2–5 minutes to load all 5 models.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

App opens at `http://localhost:5173`.

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check which models are loaded |
| `POST` | `/classify/text` | Classify a single comment |
| `POST` | `/classify/file` | Upload file for batch classification |
| `GET` | `/classify/status/{job_id}` | Poll batch job progress |

**Single comment request:**
```json
POST /classify/text
{ "text": "your comment here" }
```

**Response includes:**
```json
{
  "sentiment": "Negative",
  "sentiment_confidence": { "positive": 0.00, "neutral": 0.21, "negative": 0.79 },
  "is_uncertain": false,
  "comment_type": "Complaint",
  "toxicity": 0.001,
  "is_toxic": false,
  "is_sarcastic": true,
  "sarcasm_score": 0.87,
  "emotions": [{ "label": "admiration", "score": 0.90 }],
  "is_english": true,
  "word_analysis": [...],
  "word_counts": { "total": 13, "positive": 2, "neutral": 11, "negative": 0 },
  "multi_sentence": null,
  "latency_ms": 1465
}
```

**Rate limit:** 60 requests/minute per IP. **Max text:** 8,192 characters.

---

## How the Pipeline Works

```
Input text
    │
    ├─ Preprocessing (emoji → text, slang expansion, spell correction, URL removal)
    ├─ Gibberish detection (keyboard mash / numeric spam → early exit)
    ├─ Language detection
    │
    ├─ [Model 1] Sentiment scoring
    ├─ [Model 2] Sarcasm check → flip sentiment if ironic
    ├─ [Model 3] Toxicity scoring
    ├─ [Model 4] Emotion detection (top 5 of 28)
    ├─ [Model 5] Comment type (zero-shot + heuristic + emotion boosting)
    │
    ├─ Confidence thresholding (< 55% → "low confidence" flag)
    ├─ Multi-sentence aggregation (mixed sentiment detection)
    └─ VADER word-level analysis
```

See [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md) for a full technical breakdown of each model and pipeline step.

---

## Accuracy

Evaluated on 25 hand-labeled test cases covering slang, sarcasm, gibberish, spam, emojis, mixed sentiment, and edge cases:

| Metric | Score |
|---|---|
| Sentiment | 84% |
| Comment type | 92% |
| Combined | **88%** |

Baseline (single model, no preprocessing): ~70% combined.

---

## Project Structure

```
scc/
├── backend/
│   ├── main.py              # FastAPI app — full classification pipeline
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx           # State management, API calls
    │   ├── index.css         # Design system tokens
    │   └── components/
    │       ├── NavBar.jsx
    │       ├── ModeToggle.jsx
    │       ├── TextInput.jsx
    │       ├── FileUpload.jsx
    │       ├── SingleResult.jsx   # Result card (sentiment, emotions, words)
    │       ├── BatchResults.jsx   # Table + charts for file uploads
    │       └── Footer.jsx         # Rotating model info
    └── vite.config.js
```

---

## Supported File Formats (Batch Mode)

| Format | Notes |
|---|---|
| `.txt` | One comment per line |
| `.csv` | Auto-detects single-column files; prompts column selection for multi-column |
| `.xlsx` | Same as CSV |

Max file size: **10 MB**. Max rows: **5,000**.
