<div align="center">

# ⚡ Smart Comment Classification

### _AI-powered sentiment analysis that actually slaps_

<br />

![ModernBERT](https://img.shields.io/badge/ModernBERT-State_of_the_Art-blueviolet?style=for-the-badge&logo=huggingface&logoColor=white)
![React](https://img.shields.io/badge/React_19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Vite](https://img.shields.io/badge/Vite_8-646CFF?style=for-the-badge&logo=vite&logoColor=white)

<br />

**Drop a comment. Get the vibe. Instantly.**

_Classify thousands of comments into **Positive** / **Negative** / **Neutral** — powered by a fine-tuned ModernBERT transformer with >90% accuracy._

<br />

---

</div>

<br />

## 🧠 What is this?

A production-grade web app that takes any text comment (or an entire CSV/XLSX file of them) and tells you the sentiment — with confidence scores, batch processing, and beautiful visualizations. No more manually reading 5,000 rows of customer feedback.

> **TL;DR** — Drag. Drop. Classify. Download. Ship it. 🚀

<br />

## 🔥 Features

| | Feature | Description |
|---|---|---|
| ✍️ | **Single Comment Mode** | Type or paste any comment → instant sentiment + confidence breakdown |
| 📁 | **Bulk File Upload** | Drag & drop `.csv`, `.txt`, `.xlsx` files (up to 10MB / 5,000 rows) |
| 📊 | **Real-time Results** | Animated result cards with color-coded labels and confidence bars |
| 📋 | **Batch Results Table** | Paginated, searchable, filterable table for bulk classifications |
| ⬇️ | **CSV Export** | One-click download of all classified results |
| 🔄 | **Mode Toggle** | Seamlessly switch between text input and file upload |
| ⚡ | **Blazing Fast** | <200ms per comment on GPU, <2s on CPU |
| 🎨 | **Dark Mode Glassmorphism** | Sleek, modern UI that looks like it costs $50K |

<br />

## 🏗️ Architecture

```
┌─────────────────┐         ┌─────────────────────┐
│                 │  REST   │                     │
│   React + Vite  │ ◄─────► │  FastAPI + Uvicorn  │
│   (Frontend)    │  API    │    (Backend)        │
│                 │         │                     │
└─────────────────┘         └────────┬────────────┘
                                     │
                            ┌────────▼────────────┐
                            │                     │
                            │  ModernBERT (2024)  │
                            │  Fine-tuned Model   │
                            │  HuggingFace 🤗     │
                            │                     │
                            └─────────────────────┘
```

<br />

## 🛠️ Tech Stack

<table>
<tr>
<td><b>Layer</b></td>
<td><b>Tech</b></td>
</tr>
<tr>
<td>🖥️ Frontend</td>
<td>React 19 · Vite 8 · Axios · Recharts · Lucide Icons · Framer Motion</td>
</tr>
<tr>
<td>⚙️ Backend</td>
<td>FastAPI · Python 3.11 · Uvicorn (ASGI)</td>
</tr>
<tr>
<td>🧠 Model</td>
<td>ModernBERT-base (fine-tuned) · HuggingFace Transformers · PyTorch</td>
</tr>
<tr>
<td>📄 File Parsing</td>
<td>PapaParse (CSV) · SheetJS (XLSX) · pandas + openpyxl (server)</td>
</tr>
<tr>
<td>🚀 Deploy</td>
<td>Vercel / Netlify (frontend) · Render / HF Spaces (backend)</td>
</tr>
</table>

<br />

## ⚡ Quick Start

### Prerequisites

- **Node.js** 18+
- **Python** 3.11+
- **pip** / **venv**

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

> Frontend runs on `http://localhost:5173` · Backend on `http://localhost:8000`

<br />

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/classify/text` | Classify a single comment |
| `POST` | `/classify/file` | Upload & classify a file (CSV/TXT/XLSX) |
| `GET` | `/classify/status/{job_id}` | Poll bulk classification progress |
| `GET` | `/health` | Health check + model info |

<details>
<summary><b>Example Request & Response</b></summary>

```bash
curl -X POST http://localhost:8000/classify/text \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

```json
{
  "label": "Positive",
  "confidence": {
    "positive": 0.94,
    "neutral": 0.04,
    "negative": 0.02
  },
  "latency_ms": 83
}
```

</details>

<br />

## 📈 Model Performance

| Metric | Target |
|--------|--------|
| Test Accuracy | >90% |
| Macro F1 Score | >0.88 |
| Precision | >0.88 |
| Recall | >0.88 |
| Single Comment Latency (GPU) | <200ms |
| Bulk 500 rows (GPU) | <30s |

<br />

## 🗂️ Project Structure

```
scc/
├── frontend/               # React + Vite app
│   ├── src/
│   │   ├── components/     # NavBar, TextInput, FileUpload, Results...
│   │   ├── App.jsx         # Main app shell
│   │   ├── App.css         # Global styles
│   │   └── main.jsx        # Entry point
│   ├── public/             # Static assets
│   └── package.json
├── backend/                # FastAPI server (Python)
│   ├── main.py             # API routes
│   ├── model/              # Fine-tuned ModernBERT weights
│   └── requirements.txt
└── PRD.md                  # Product Requirements Document
```

<br />

## 👥 Team

| Name | Role |
|------|------|
| **V. Raghavendra Goud** | Developer |
| **R. Rechal** | Developer |
| **R. Srinivas** | Developer |
| **Dr. K. Rameshwaraiah** | Project Guide |

<br />

## 📜 License

Built with ❤️ and caffeine for Batch C1 — March 2026

<div align="center">

---

<sub>Powered by **ModernBERT** · Built with **React** + **FastAPI** · Designed to impress</sub>

</div>
