<div align="center">

# ⚡ Smart Comment Classification

### AI-powered comment analysis with a multi-model NLP pipeline

<br />

[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-blueviolet?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![React](https://img.shields.io/badge/React_19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Vite](https://img.shields.io/badge/Vite_8-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vite.dev/)

<br />

> **Drop a comment. Get the vibe. Instantly.** ✨

Classify comments into **Positive** / **Negative** / **Neutral** and enrich them with **type**, **toxicity**, **emotion**, and **sarcasm** signals using a production-oriented ensemble of Hugging Face models plus lightweight heuristics.

</div>

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| ✍️ | Single Comment | Type or paste → instant sentiment + confidence |
| 📁 | Bulk Upload | Drag & drop `.csv`, `.txt`, `.xlsx` (10MB / 5,000 rows) |
| 📊 | Real-time Results | Animated cards with color-coded labels |
| 📋 | Batch Table | Paginated, searchable, filterable results |
| ⬇️ | CSV Export | One-click download of all classified results |
| 🔄 | Mode Toggle | Switch between text input & file upload |
| ⚡ | Blazing Fast | <200ms/comment on GPU, <2s on CPU |
| 🎨 | Dark Glassmorphism | Sleek, modern UI |
| 🧪 | Sentiment | Positive, Neutral, Negative with confidence |
| 📝 | Comment Type | Praise, Complaint, Question, Feedback, Spam |
| ☠️ | Toxicity | Scores abusive language 0–100% |
| 🎭 | Emotions | 28 fine-grained emotions |
| 😏 | Sarcasm | Catches ironic positivity |
| 🔍 | Word Highlighting | Color-coded words by sentiment |
| 📖 | Multi-sentence | Detects mixed sentiment |
| 🚫 | Gibberish Filter | Rejects keyboard mashing |
| 🗣️ | Informal English | Handles slang, emojis |

---

## 🏗️ Architecture

```mermaid
graph LR
    A[React + Vite<br/>Frontend] -->|REST API| B[FastAPI<br/>Backend]
    B --> C[Sentiment + Sarcasm + Toxicity<br/>Zero-shot Type + Emotions]
    C --> D[HuggingFace<br/>🤗 Models + Heuristics]
    
    style A fill:#61DAFB,stroke:#333,color:#000
    style B fill:#009688,stroke:#333,color:#fff
    style C fill:#blueviolet,stroke:#333,color:#fff
    style D fill:#ff6b6b,stroke:#333,color:#fff
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|:---:|:---|
| 🖥️ Frontend | React 19 · Vite 8 · Axios · Recharts · Framer Motion |
| ⚙️ Backend | FastAPI · Python 3.11 · Uvicorn |
| 🧠 ML/AI | RoBERTa · BERT · BART MNLI · GoEmotions · VADER · HuggingFace Transformers · PyTorch |
| 📄 File Parsing | PapaParse · SheetJS · pandas |

---

## 🤖 Models

| Model | Task | Params |
|:---|:---:|:---:|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment | 125M |
| `unitary/toxic-bert` | Toxicity | 110M |
| `facebook/bart-large-mnli` | Comment Type | 400M |
| `SamLowe/roberta-base-go_emotions` | Emotions | 125M |
| `cardiffnlp/twitter-roberta-base-irony` | Sarcasm | 125M |
| VADER Lexicon | Word Sentiment | Rule-based |

> 📦 First run downloads multiple model weights from Hugging Face

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+
- **Python** 3.11+

### Backend

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

> 🖥️ Backend: http://localhost:8000  
> ⏱️ First startup: 2–5 minutes to load all 5 models

### Frontend

```bash
cd frontend
npm install
npm run dev
```

> 🌐 Frontend: http://localhost:5173

---

## 🔌 API Reference

| Method | Endpoint | Description |
|:---:|:---|:---|
| GET | /health | Health check + model info |
| POST | /classify/text | Classify single comment |
| POST | /classify/file | Upload file for batch |
| GET | /classify/status/{job_id} | Poll batch progress |

### Example Request

```bash
curl -X POST http://localhost:8000/classify/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

### Example Response

```json
{
  "sentiment": "Positive",
  "sentiment_confidence": {
    "positive": 0.94,
    "neutral": 0.04,
    "negative": 0.02
  },
  "comment_type": "Praise",
  "toxicity": 0.001,
  "is_toxic": false,
  "is_sarcastic": false,
  "emotions": [{"label": "admiration", "score": 0.90}],
  "latency_ms": 83
}
```

> 📝 **Rate Limit:** 60 requests/minute | **Max Text:** 8,192 chars

---

## 📈 Performance

| Metric | Score |
|:---|:---:|
| 🎯 Accuracy | Not yet tracked by an automated eval suite |
| 📊 F1 Score | Not yet tracked by an automated eval suite |
| ⚡ Single Comment | Depends on which models are loaded and hardware |
| 📦 Bulk Processing | Batched in the backend as of v3.1 |

---

## ⚙️ How It Works

```mermaid
flowchart TD
    A[Input Text] --> B[Preprocessing]
    B --> C[Gibberish Detection]
    C --> D{Language Check}
    D -->|Non-English| E[Flag Lower Confidence]
    D -->|English| F[Model Pipeline]
    
    F --> G[1️⃣ Sentiment]
    G --> H[2️⃣ Sarcasm Check]
    H --> I{Is Sarcastic?}
    I -->|Yes| J[Flip Sentiment]
    I -->|No| K[3️⃣ Toxicity]
    K --> L[4️⃣ Emotions]
    L --> M[5️⃣ Comment Type]
    
    J --> N[Post-processing]
    K --> N
    M --> N
    N --> O[Confidence Threshold]
    O --> P[Word-level VADER]
    P --> Q[Final Output]
    
    style A fill:#f9f,stroke:#333
    style Q fill:#9f9,stroke:#333
    style E fill:#f99,stroke:#333
```

---

## 📂 Supported Files

| Format | Description |
|:---|:---|
| 📄 .txt | One comment per line |
| 📊 .csv | Auto-detects column; prompts if multi-column |
| 📗 .xlsx | Excel files |

> ⚠️ **Max file:** 10 MB | **Max rows:** 5,000

---

## Current Architecture

This project is **not** running ModernBERT in production today.

The live backend in `backend/main.py` currently uses:

- `cardiffnlp/twitter-roberta-base-sentiment-latest` for sentiment
- `cardiffnlp/twitter-roberta-base-irony` for sarcasm / irony
- `unitary/toxic-bert` for toxicity
- `SamLowe/roberta-base-go_emotions` for emotion tags
- `facebook/bart-large-mnli` for zero-shot comment type classification
- VADER for word-level lexical highlighting

Recent hardening work added:

- Tokenizer-aware truncation instead of raw character slicing
- Explicit model-load status and degraded health reporting
- Real backend batching for file jobs
- Regression tests for preprocessing, truncation, sarcasm, and batching
- Stage-level telemetry and heuristic/truncation metadata in responses

## ModernBERT Support

ModernBERT is now supported as the **preferred sentiment backend** when you provide a fine-tuned checkpoint path or Hugging Face repo through the `MODERNBERT_SENTIMENT_MODEL` environment variable.

Example:

```bash
set MODERNBERT_SENTIMENT_MODEL=your-org/modernbert-comment-sentiment
```

Important:

- The repo still does **not** contain a fine-tuned ModernBERT sentiment model by default.
- If `MODERNBERT_SENTIMENT_MODEL` is missing or fails to load, the backend falls back to `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- The app UI and `/health` endpoint now expose which sentiment backend is actually active.
- A training scaffold for producing that checkpoint now lives in `backend/training/train_modernbert_sentiment.py`.
- The training script can now pull `cardiffnlp/tweet_eval` directly from Hugging Face for a fast public-data training path.
- Full setup instructions live in `docs/MODERNBERT_SETUP.md`.

## 📁 Project Structure

```
scc/
├── frontend/
│   ├── src/
│   │   ├── components/     # NavBar, TextInput, FileUpload...
│   │   ├── App.jsx         # Main app shell
│   │   └── main.jsx        # Entry point
│   └── package.json
├── backend/
│   ├── main.py             # API routes + ML pipeline
│   └── requirements.txt
└── docs/
    ├── HOW_IT_WORKS.md
    └── PRD.md
```

---

## 📜 License

<div align="center">

Made with ❤️ and caffeine — March 2026

---

**Powered by ModernBERT · Built with React + FastAPI**

</div>
