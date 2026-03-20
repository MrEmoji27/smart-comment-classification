# Smart Comment Classification (SCC)
## Complete Technical Documentation — Part 1 of 3

**Project:** Smart Comment Classification
**Version:** 3.2
**Stack:** React 19 + FastAPI + Python 3.11 + HuggingFace Transformers
**Model:** ModernBERT (fine-tuned) / Twitter RoBERTa (fallback)
**Team:** V. Raghavendra Goud · R. Rechal · R. Srinivas
**Guide:** Dr. K. Rameshwaraiah
**Batch:** C1 — March 2026

---

## Part 1 Contents

- Section 1 — Executive Summary
- Section 2 — Project Background and Motivation
- Section 3 — System Architecture Overview
- Section 4 — Technology Stack Deep Dive
- Section 5 (Parts A–D) — Backend Architecture:
  - Application Initialization
  - Rate Limiting
  - Language Detection
  - Spell Correction
  - Text Preprocessing Pipeline
  - Gibberish Detection
  - Multi-Sentence Splitting
  - Model Loading and Registry
  - Tokenizer-Aware Truncation
  - Batch Inference Engine

---

## 1. Executive Summary

**Smart Comment Classification (SCC)** is a production-grade, AI-powered web application designed to analyze and categorize textual comments across six simultaneous analytical dimensions. Every comment processed by SCC yields:

1. **Sentiment** — Positive, Neutral, or Negative, with a three-way probability distribution.
2. **Comment Type** — Praise, Question, Feedback, Complaint, Spam, or Other.
3. **Toxicity Score** — A continuous 0–1 probability of abusive language.
4. **Emotion Tags** — Up to 5 of 28 fine-grained emotional labels.
5. **Sarcasm Detection** — Whether the surface sentiment conflicts with the implied sentiment.
6. **Word-Level Sentiment** — Per-token polarity coloring using VADER lexical analysis.

The application supports two modes of operation: real-time single comment analysis (sub-200ms on GPU) through a text input interface, and asynchronous bulk file processing (CSV, TXT, XLSX — up to 10 MB / 5,000 rows) through a drag-and-drop file upload interface.

The classification pipeline is a staged ensemble of five transformer models and one lexical tool:

| Stage | Model | Parameters | Hosting |
|---|---|---|---|
| 1 — Sentiment | ModernBERT (fine-tuned) or Twitter RoBERTa | 149M / 125M | Local / HuggingFace Hub |
| 2 — Sarcasm | Twitter RoBERTa Irony (Cardiff NLP) | 125M | HuggingFace Hub |
| 3 — Toxicity | Toxic-BERT (Jigsaw) | 110M | HuggingFace Hub |
| 4 — Emotions | GoEmotions RoBERTa (SamLowe) | 125M | HuggingFace Hub |
| 5 — Comment Type | BART-large MNLI (Facebook) | 400M | HuggingFace Hub |
| 6 — Word-level | VADER Lexicon | Rule-based | Python package |

All six stages run in sequence per comment request, with batching across items in a file job to maximise GPU throughput.

The system includes a fully functional training scaffold (`backend/training/train_modernbert_sentiment.py`) that enables teams to fine-tune a ModernBERT checkpoint on their own labelled dataset and hot-swap it into the production pipeline with zero code changes via an environment variable.

The frontend is a React 19 single-page application with glassmorphism dark-mode design, animated result visualisations (anime.js timeline animations, Framer Motion page transitions), a paginated sortable results table (BatchResults), a Recharts donut chart for sentiment distribution, and CSV export functionality.

---

## 2. Project Background and Motivation

### 2.1 Problem Statement

Online platforms generate immense volumes of user-generated textual content daily:

- E-commerce platforms receive thousands of product reviews per minute.
- Customer support portals accumulate complaint tickets that must be triaged by severity.
- Social media monitoring teams need real-time brand sentiment tracking.
- App stores display millions of user ratings and reviews.
- Educational platforms accumulate student feedback on courses and instructors.

Manually reviewing this volume is economically infeasible. Hiring human annotators for sentiment analysis at scale costs thousands of dollars per day for even modest data volumes, introduces inter-annotator disagreement, and cannot operate in real time.

### 2.2 Why Legacy Approaches Fail

**Keyword matching and regular expressions** were the first generation of automated sentiment tools. A keyword matcher might flag any comment containing "bad" as Negative. But consider:

> "Not bad at all, really exceeded my expectations!"

A naive keyword match on "bad" misclassifies this as Negative. The negation "Not bad" is a common positive idiom in English, and the trailing phrase is clearly positive. Regular expressions cannot capture semantic context.

**Traditional machine learning** (Naive Bayes, SVM with TF-IDF features) improves on keyword matching by learning statistical associations between n-grams and labels. However, TF-IDF features:

- Cannot capture word order beyond n-grams (no long-range dependencies).
- Require large manually-labelled training sets to generalise.
- Do not share knowledge across different domains.
- Fail on out-of-vocabulary words (new slang, brand names).

**Pre-transformer deep learning** (LSTM, BiLSTM, CNN text classifiers) introduced sequential context through recurrence, but still struggled with:

- Very long-range dependencies (sentiment of a 500-word review depends on its opening sentence).
- Transfer learning was limited — each task required full retraining from scratch.
- Attention mechanisms were added but not as efficient as the full self-attention of transformers.

### 2.3 The Transformer Revolution

The **Transformer architecture** (Vaswani et al., 2017) introduced fully parallelisable self-attention, allowing every word to attend to every other word in a single pass. BERT (Devlin et al., 2018) demonstrated that a bidirectional transformer pre-trained on a large unlabelled corpus via Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) could achieve state-of-the-art results on 11 NLP tasks with minimal fine-tuning.

This paradigm — **pre-train on large generic corpus, fine-tune on domain-specific task** — revolutionised NLP by enabling high accuracy with relatively small labelled datasets.

### 2.4 SCC's Unique Challenges

SCC is specifically designed for **informal English comment text**, which has characteristics that make it harder than formal text:

| Challenge | Example | SCC Solution |
|---|---|---|
| Slang and abbreviations | "smh this slaps ngl" | SLANG_MAP expansion |
| Emoji | "" | emoji.demojize() |
| Contractions | "can't", "shouldn't" | CONTRACTIONS expansion |
| Repeated characters | "sooooo good" | regex deduplication |
| Hashtags | "#ThisIsBad" | CamelCase segmentation |
| Sarcasm | "Oh great, crashed again" | irony model + context rules |
| Gibberish/spam | "qwerty asdfgh" | vowel ratio + keyboard seq detection |
| Code-mixing | "yaar this product is bahat acha" | English ratio check with low threshold |
| Multi-sentence mixed sentiment | "Good design. Terrible battery. Ok overall." | sentence-level aggregation |

### 2.5 ModernBERT: State-of-the-Art Encoder

**ModernBERT** (answerdotai/ModernBERT-base, released December 2024) represents the current state-of-the-art in encoder-only transformers. It addresses the main limitations of BERT/RoBERTa:

**Architecture improvements over BERT:**

| Feature | BERT | RoBERTa | ModernBERT |
|---|---|---|---|
| Max context | 512 tokens | 512 tokens | 8,192 tokens |
| Position encoding | Absolute sinusoidal | Absolute learned | RoPE (Rotary) |
| Attention | Full O(n²) | Full O(n²) | Alternating local+global |
| Activation | GeLU | GeLU | GeGLU |
| Pre-training tokens | 3.3B | 16B | 2,000B |
| Architecture layers | 12 | 12 | 22 |
| NSP objective | Yes | No | No |
| Flash Attention | No | No | Yes (v2) |

**RoPE (Rotary Position Embeddings)**: Encodes position as a rotation of the query/key vectors in the attention mechanism. This means position information is relative (how far apart are two tokens) rather than absolute (what position is this token). RoPE generalises better to sequence lengths not seen during training, enabling ModernBERT's 8K context window.

**Alternating attention**: Every third encoder layer uses global (full) attention where every token attends to every other. The other two-thirds of layers use local sliding-window attention with a window of 128 tokens. This reduces the computational cost from O(n²) to O(n × 128) for most layers while maintaining full-context awareness through periodic global layers.

**Impact for SCC**: Long product reviews, forum posts, or multi-paragraph feedback (which could easily exceed 512 tokens) can be processed by ModernBERT without any truncation, preserving the full semantic context of the entire comment.

### 2.6 Project Scope and Deliverables

The SCC project was scoped to deliver the following artifacts:

1. **Backend API** (`backend/main.py`): FastAPI RESTful service with five-stage NLP pipeline.
2. **Frontend SPA** (`frontend/src/`): React 19 application with full feature set.
3. **Training pipeline** (`backend/training/train_modernbert_sentiment.py`): Reproducible fine-tuning script.
4. **Fine-tuned model** (`backend/models/modernbert-sentiment/`): Trained checkpoint with metadata.
5. **Test suite** (`backend/tests/test_backend.py`): 6 regression tests.
6. **Documentation suite** (`docs/`): PRD, HOW_IT_WORKS, MODERNBERT_SETUP, Technical Documentation.
7. **Product demo video** (`video/`): Remotion-rendered 30-second product demonstration.

---

## 3. System Architecture Overview

### 3.1 Conceptual Architecture

SCC follows a classic **two-tier architecture**:

- **Tier 1 — Presentation Layer**: React SPA served via Vite dev server (development) or as static files (production).
- **Tier 2 — Application + Data Layer**: FastAPI backend embedding the full ML inference pipeline.

There is no dedicated database tier. Persistence is handled by:
- In-memory `jobs` dict for batch job tracking (ephemeral).
- HuggingFace model cache on the local filesystem (~/.cache/huggingface/).
- Fine-tuned model stored in `backend/models/modernbert-sentiment/`.

### 3.2 Component Diagram

```
+============================================================+
|                    User's Browser                          |
|  +-----------------+    +--------------------------------+ |
|  |    TextInput    |    |          FileUpload            | |
|  |    Component    |    |          Component             | |
|  +-----------------+    +--------------------------------+ |
|          |                         |                       |
|          v                         v                       |
|  +-----------------+    +--------------------------------+ |
|  |  SingleResult   |    |         BatchResults           | |
|  |  Component      |    |         Component              | |
|  | (anime.js)      |    | (Recharts, XLSX export)        | |
|  +-----------------+    +--------------------------------+ |
|                                                            |
|  Root: App.jsx (Framer Motion, Mantine, Axios)             |
+============================================================+
                 |    Axios HTTP/JSON    |
                 v                      v
+============================================================+
|                  FastAPI Backend (Port 8000)               |
|  +-------------------+   +-----------------------------+  |
|  | RateLimiter       |   | TextPreprocessor            |  |
|  | (sliding window)  |   | (emoji, slang, contractions)|  |
|  +-------------------+   +-----------------------------+  |
|                                                            |
|  +-------------------+   +-----------------------------+  |
|  | GibberishDetector |   | ModelRegistry               |  |
|  | (vowel/consonant) |   | (load, fallback, status)    |  |
|  +-------------------+   +-----------------------------+  |
|                                                            |
|  +------------------------------------------------------+  |
|  |            5-Stage ML Inference Pipeline             |  |
|  |  Stage 1: Sentiment  --> _normalize_sentiment()      |  |
|  |  Stage 2: Sarcasm    --> _apply_sarcasm_adjustment() |  |
|  |  Stage 3: Toxicity   --> extract_label_score()       |  |
|  |  Stage 4: Emotions   --> top-5 emotions              |  |
|  |  Stage 5: Type       --> apply_type_heuristics()     |  |
|  +------------------------------------------------------+  |
|                                                            |
|  +-------------------+   +-----------------------------+  |
|  | VADER Word Anal.  |   | Job Queue (in-memory)       |  |
|  | (per-token score) |   | (batch file processing)     |  |
|  +-------------------+   +-----------------------------+  |
+============================================================+
                            |
                            v
+============================================================+
|         HuggingFace Transformers Model Stack               |
|                                                            |
|  ModernBERT-base fine-tuned  [LOCAL]                       |
|  cardiffnlp/twitter-roberta-base-sentiment-latest          |
|  cardiffnlp/twitter-roberta-base-irony                     |
|  unitary/toxic-bert                                        |
|  SamLowe/roberta-base-go_emotions                          |
|  facebook/bart-large-mnli                                  |
|  VADER Lexicon (vaderSentiment package)                    |
+============================================================+
```

### 3.3 Request Processing — Single Comment

The single comment classification flow traverses these stages in order:

**Stage 0 — Gateway**
- `POST /classify/text` received.
- `check_rate_limit(client_ip)` — raises HTTP 429 if IP exceeds 60 req/min.
- Input validation: non-empty, <= 8192 characters.
- `_ensure_core_models()` — raises HTTP 503 if required models unavailable.

**Stage 1 — Text Processing**
- `preprocess_text(text)` — emoji demojise, URL/mention removal, hashtag segmentation, character dedup, spell correction, contraction/slang expansion.
- `is_gibberish(text)` — if True, all model stages are skipped and a fixed Neutral/Spam response is returned.
- `detect_language_is_english(text)` — sets the `is_english` flag.
- Per-model `truncate_for_model(cleaned, model_name)` — tokeniser-aware truncation for each of the 5 models.

**Stage 2 — Sentiment**
- `_run_stage_batch("sentiment", [text])` — runs the sentiment model.
- `_normalize_sentiment_output(result)` — normalises label names to Positive/Neutral/Negative.
- `predicted_sentiment = max(sent_scores, key=sent_scores.get)`.

**Stage 3 — Sarcasm**
- `_run_stage_batch("sarcasm", [text])` — runs the irony model.
- `_extract_label_score(result, {"irony", "label_1", "1"})` — extracts irony probability.
- `_apply_sarcasm_adjustment(record, ...)` — if irony > 0.80 AND negative context words present, flips Positive sentiment to Negative.

**Stage 4 — Toxicity**
- `_run_stage_batch("toxicity", [text])`.
- `_extract_label_score(result, {"toxic"})` — extracts toxicity probability.
- `is_toxic = tox_score > 0.5`.

**Stage 5 — Emotions**
- `_run_stage_batch("emotion", [text])`.
- Top-5 emotions returned directly.

**Stage 6 — Comment Type**
- `_run_stage_batch("type", [text], candidate_labels=CANDIDATE_TYPES)`.
- `apply_type_heuristics(text, raw_type_scores, sentiment, tox, emotions)` — adjusts scores with 9 heuristic rules.
- `apply_confidence_flag(sentiment, sent_scores)` — sets `is_uncertain` if max confidence < 0.55.

**Stage 7 — Aggregation**
- `classify_multi_sentence(text, cleaned)` — if 2+ sentences detected, re-classify each individually and average scores.
- `analyze_word_sentiment(text)` — VADER word-level sentiment for every token.

**Stage 8 — Response Assembly**
- Build the full response dict with all signals, timings, metadata.
- Return `JSONResponse(content=result)`.

### 3.4 Request Processing — File Upload

File processing uses FastAPI's `BackgroundTasks` mechanism:

1. The HTTP handler parses the file, creates a job entry, adds `process_batch_job` as a background task, and returns `{ job_id, status: "processing", total_rows }` immediately.
2. FastAPI sends the HTTP response to the client.
3. After the response is sent, the ASGI event loop runs `process_batch_job(job_id, texts)`.
4. `process_batch_job` iterates through texts in batches of 16, calling `classify_texts_internal()` for each batch. The `job["processed"]` counter is updated after each item.
5. When all texts are processed, `job["status"] = "done"` and `job["results"] = [...]`.
6. The frontend polls `GET /classify/status/{job_id}` every 1 second to check progress. When `status == "done"`, it reads the results and renders the BatchResults component.

### 3.5 ModernBERT Integration Architecture

The sentiment model slot supports two candidates in a priority order:

```
Primary:  ModernBERT fine-tuned checkpoint
          -> Detected via MODERNBERT_SENTIMENT_MODEL env var
             OR via backend/models/modernbert-sentiment/config.json

Fallback: cardiffnlp/twitter-roberta-base-sentiment-latest
          -> Always available from HuggingFace Hub
          -> Used automatically if ModernBERT fails to load
```

This design means the application always starts successfully — it degrades gracefully to the fallback rather than refusing to start.

### 3.6 Directory Structure

```
scc/
├── README.md                         # Project overview, quick start
├── test_comments.csv                 # Sample test data
│
├── backend/
│   ├── main.py                       # FastAPI app, all pipeline logic
│   ├── requirements.txt              # Python dependencies
│   ├── models/
│   │   └── modernbert-sentiment/     # Fine-tuned ModernBERT checkpoint
│   │       ├── config.json
│   │       ├── model.safetensors
│   │       ├── tokenizer.json
│   │       ├── tokenizer_config.json
│   │       ├── training_args.bin
│   │       ├── scc_model_metadata.json
│   │       ├── checkpoint-2852/      # Intermediate checkpoint
│   │       └── checkpoint-4278/      # Latest checkpoint
│   ├── training/
│   │   └── train_modernbert_sentiment.py
│   └── tests/
│       └── test_backend.py
│
├── frontend/
│   ├── index.html                    # SPA entry point
│   ├── vite.config.js                # Vite build configuration
│   ├── package.json                  # NPM dependencies
│   ├── src/
│   │   ├── main.jsx                  # React root render
│   │   ├── App.jsx                   # Root component, state machine
│   │   ├── App.css                   # Global layout styles
│   │   ├── index.css                 # CSS custom properties (theme)
│   │   └── components/
│   │       ├── NavBar.jsx/.css
│   │       ├── TextInput.jsx/.css
│   │       ├── FileUpload.jsx/.css
│   │       ├── SingleResult.jsx/.css
│   │       ├── BatchResults.jsx/.css
│   │       ├── Footer.jsx/.css
│   │       └── ModeToggle.jsx/.css
│   └── dist/                         # Production build output
│
├── docs/
│   ├── PRD.md                        # Product Requirements Document
│   ├── HOW_IT_WORKS.md               # User-facing explanation
│   ├── MODERNBERT_SETUP.md           # ModernBERT integration guide
│   └── TECHNICAL_DOCUMENTATION_PART*.md  # This document
│
└── video/
    ├── src/ProductDemo.jsx           # Remotion video script
    └── out/product-demo.mp4          # Rendered demo
```

---

## 4. Technology Stack Deep Dive

### 4.1 Backend Technologies

#### 4.1.1 FastAPI 0.115.0

FastAPI is a modern, high-performance Python web framework built on top of Starlette (ASGI) and Pydantic (data validation). It was chosen for SCC for several reasons:

**Async-native design**: FastAPI is built for ASGI (Asynchronous Server Gateway Interface), meaning it can handle many concurrent HTTP connections without blocking. While ML inference itself is CPU-bound and blocks the event loop, the async design allows health checks, file parsing, and job status queries to remain responsive.

**Type hints and automatic documentation**: FastAPI auto-generates OpenAPI (Swagger) documentation from Python type annotations. While SCC uses raw `Request` objects (for rate limiting flexibility), the framework still provides auto-docs at `/docs`.

**Dependency injection**: FastAPI's `Depends()` system allows clean separation of concerns — rate limiting, model availability checks, and request parsing can each be a separate dependency.

**Background tasks**: `BackgroundTasks.add_task()` schedules a function to run after the HTTP response has been sent. This is the mechanism SCC uses for batch file processing — the user gets an immediate `job_id` while inference runs asynchronously.

**CORS middleware**: `CORSMiddleware` from Starlette is added as a middleware layer to allow the React frontend (different origin) to make API calls.

#### 4.1.2 Uvicorn 0.30.6

Uvicorn implements the ASGI interface as an HTTP server. It wraps the FastAPI app and handles:

- **TCP connection management**: Accepts incoming connections, manages keep-alive.
- **HTTP/1.1 and HTTP/2**: Uvicorn supports both protocols.
- **WebSocket support**: Available but not used in SCC.
- **Reload mode**: `--reload` flag watches for file changes and restarts the server automatically during development.

**Single-worker limitation**: By default Uvicorn runs one worker process. For CPU-bound tasks like ML inference, multiple workers do not help (Python GIL limits true CPU parallelism in threads) — they would each load all models into memory separately, multiplying VRAM/RAM usage. The correct scaling strategy is GPU batching, not multiple workers.

#### 4.1.3 PyTorch 2.4.1+

PyTorch is the foundation of all HuggingFace transformer models in SCC. Key concepts:

**Tensors**: PyTorch's multi-dimensional array type, analogous to NumPy arrays but GPU-accelerable. Tokenized text becomes a 2D tensor of shape `[batch_size, sequence_length]`.

**`torch.cuda.is_available()`**: Checks if a CUDA-capable GPU is present and drivers are installed. Returns `True` on machines with NVIDIA GPU + CUDA toolkit + PyTorch CUDA build. SCC uses this to set `PIPELINE_DEVICE = 0` (GPU) or `-1` (CPU).

**Autograd (not used in inference)**: PyTorch tracks gradients through operations for backpropagation during training. During inference (`torch.no_grad()` context, which HuggingFace pipelines set automatically), gradient tracking is disabled for performance.

**`model.safetensors` format**: ModernBERT's weights are saved in the safetensors format, which stores raw tensor data without pickle serialisation. This prevents arbitrary Python code execution during model loading — a security concern with the older `.bin` format.

#### 4.1.4 HuggingFace Transformers 4.48.0+

The `transformers` library abstracts the complexity of working with pre-trained models:

**`pipeline(task, model, tokenizer, device)`**: High-level API. For a `"sentiment-analysis"` pipeline:
1. Tokenises input text using the model's tokeniser (BPE or WordPiece).
2. Converts token IDs to a PyTorch tensor.
3. Runs the tensor through the model.
4. Applies softmax to logits to get probabilities.
5. Returns `[{"label": "...", "score": ...}]`.

**`AutoTokenizer.from_pretrained(model_id)`**: Automatically selects the correct tokeniser class (BertTokenizerFast, RobertaTokenizerFast, etc.) based on the model's `tokenizer_config.json`. The `use_fast=True` flag uses the Rust-based tokeniser for significantly faster tokenisation.

**`AutoModelForSequenceClassification`**: Used in the training script. Loads the model architecture from `config.json` and weight from `model.safetensors`, with a classification head (linear layer) on top of the encoder's `[CLS]` token.

**Caching**: HuggingFace caches downloaded models in `~/.cache/huggingface/hub/`. On first run with a new model ID, weights are downloaded. On subsequent runs, they are loaded from cache without network access.

#### 4.1.5 VADER Sentiment 3.3.2+

**VADER** (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analysis tool specifically designed for social media text. Unlike ML models, it requires no training, no GPU, and runs in microseconds per token.

VADER's lexicon was constructed by crowdsourcing sentiment ratings for 7,517 lexical features (words, acronyms, emoticons, punctuation patterns). Each feature has a mean sentiment score on a scale from -4 (extremely negative) to +4 (extremely positive), aggregated from ~10 human raters.

**Compound score formula:**

```
sum_s = sum of all lexical scores in the text
compound = sum_s / sqrt(sum_s² + alpha)  where alpha=15
```

The division by `sqrt(sum_s² + 15)` normalises the compound to [-1, 1]. The constant 15 controls how quickly the score saturates (prevents a single very negative word from dominating a long text).

**Rule modifiers applied by VADER:**

1. **Punctuation**: Each `!` up to 3 adds 0.292 to the compound score magnitude.
2. **ALL CAPS**: Adds 0.733 to a positive word's raw score or subtracts 0.733 from a negative word's raw score.
3. **Booster words**: `"very"`, `"incredibly"`, `"extremely"` before a valenced word multiply the word's score by the booster's weight (VADER has a weighted list of ~50 boosters).
4. **Negation**: `"not"`, `"no"`, `"never"`, `"without"` flip the sign and reduce magnitude by multiplying by -0.74.
5. **But clause**: After a `"but"` in the text, the compound is recalculated with 75% weight on the post-"but" portion.

In SCC, VADER is used only for **word-level token scoring**, not for the primary sentiment label. The primary label comes from the transformer models which have far higher accuracy on informal text.

#### 4.1.6 emoji 2.8.0+

The `emoji` library converts Unicode emoji characters to their textual description based on the Unicode CLDR (Common Locale Data Repository) data:

- `emoji.demojize("", delimiters=(" ", " "))` → `" :thumbs_up: "`
- `emoji.demojize("", delimiters=(" ", " "))` → `" :red_heart: "`
- `emoji.demojize("", delimiters=(" ", " "))` → `" :face_with_tears_of_joy: "`

The space delimiters ensure the resulting text token integrates cleanly with the surrounding words when tokenised. Without demojisation, the transformer's tokeniser would split emoji into bizarre subword tokens (Unicode code point fragments) with no semantic meaning.

### 4.2 Frontend Technologies

#### 4.2.1 React 19

React 19 introduces several improvements relevant to SCC:

**`Suspense` for lazy loading**: `FileUpload` and `BatchResults` are imported with `lazy()`, meaning their JavaScript bundles are only loaded when needed:

```javascript
const FileUpload = lazy(() => import('./components/FileUpload'));
const BatchResults = lazy(() => import('./components/BatchResults'));
```

This reduces the initial JavaScript bundle size. The text input and single result components (used most frequently) are always bundled.

**`useMemo` for expensive computations**: `BatchResults` uses `useMemo` for `filteredData`, `paginatedData`, `stats`, and `advancedStats` — all of which iterate over potentially 5,000-row arrays. Without memoisation, these would recompute on every render triggered by typing in the search box.

**Concurrent rendering compatibility**: Framer Motion's `AnimatePresence` and `motion.div` work correctly with React 19's concurrent rendering model, which may pause and resume rendering for high-priority updates.

#### 4.2.2 Vite 8

Vite 8 uses **Rolldown** (a Rust-based JavaScript bundler) for production builds:

**Development mode** (`npm run dev`):
- Files are served as ES modules directly — no bundling.
- HMR (Hot Module Replacement) updates only the changed module.
- Startup is near-instant regardless of project size.

**Production mode** (`npm run build`):
- Rolldown bundles and tree-shakes the module graph.
- Code splitting: lazy-imported components become separate chunks.
- CSS extraction: component CSS files are bundled into `index-*.css`.
- Asset fingerprinting: output files are named with content hashes (e.g., `index-B-OFjtxN.js`) for cache-busting.

**`vite.config.js`** configures:
- The `@vitejs/plugin-react` plugin for JSX transformation (using the new JSX transform that doesn't require `import React`).
- The API proxy configuration (`/api` → `localhost:8000`) if used.
- The `VITE_API_URL` environment variable injection.

#### 4.2.3 Mantine UI 7+

Mantine is a React UI library with 100+ components and a comprehensive theming system. SCC uses it selectively for form elements and contextual components:

**`MantineProvider`** at the root level:
- `theme` — custom theme with blue colour scale and primary font.
- `forceColorScheme={theme}` — forces dark or light mode regardless of OS preference.

**Components used:**
- `Button` — the primary action button in FileUpload (with `loading` state, `leftSection` icon).
- `Paper` — a card container with padding and rounded corners used in FileUpload.
- `Badge` — format pills (.csv, .txt, .xlsx).
- `Select` — the column picker dropdown.
- `Text` — semantic text element with `component="label"`.
- `Alert` — the error toast notification.

Non-Mantine components (NavBar, SingleResult, BatchResults, TextInput) use custom CSS with the project's design token system.

#### 4.2.4 Framer Motion (motion/react)

Framer Motion handles declarative layout and mount/unmount transitions:

**`AnimatePresence mode="wait"`**: In "wait" mode, the exit animation of the leaving component must complete before the entering component begins its entrance animation. This prevents overlapping transitions. SCC uses this for:
- Switching between TextInput and FileUpload modes.
- Switching between SingleResult and BatchResults (or empty state).

**`motion.div` props pattern:**

```javascript
const viewTransition = {
    initial: { opacity: 0, y: 12 },
    animate: { opacity: 1, y: 0 },
    exit:    { opacity: 0, y: -8 },
    transition: { duration: 0.28, ease: [0.16, 1, 0.3, 1] }
};
```

The easing `[0.16, 1, 0.3, 1]` is a cubic Bezier approximating a spring curve — it starts slow and decelerates, giving a natural "settle" feeling as elements enter.

#### 4.2.5 anime.js

anime.js is a lightweight (~18KB) JavaScript animation library used in `SingleResult.jsx` for the result card entrance animation. Unlike Framer Motion (which handles React lifecycle) anime.js directly manipulates CSS properties via requestAnimationFrame:

```javascript
const timeline = anime.timeline({ easing: 'easeOutExpo' });

timeline
    .add({ targets: root, opacity: [0, 1], translateY: [14, 0], duration: 320 })
    .add({ targets: stagedNodes, opacity: [0, 1], translateY: [12, 0],
           duration: 420, delay: anime.stagger(55) }, '-=120')
    .add({ targets: bars, scaleX: [0, 1], duration: 560,
           delay: anime.stagger(45), easing: 'easeOutQuart' }, '-=240');
```

**Timeline composition**: Each `.add()` call adds an animation segment. The string offset `'-=120'` means "start 120ms before the previous segment ends", creating overlap for a fluid cascade effect.

**`anime.stagger(55)`**: Creates a staggered delay — the first element starts at 0ms, the second at 55ms, the third at 110ms, etc. This makes the result sections appear to "fall into place" sequentially.

**`easeOutExpo` vs `easeOutQuart`**: The card and text sections use `easeOutExpo` (fast deceleration, feels light). The confidence bars use `easeOutQuart` (slightly heavier deceleration, feels like a physical bar filling up).

**Cleanup in `useEffect`**:
```javascript
return () => {
    timeline.pause();
    anime.remove(root);
    anime.remove(stagedNodes);
    anime.remove(bars);
};
```

This cancels active animations and removes anime.js's references when the component unmounts or the result changes, preventing memory leaks and animation conflicts.

---

## 5. Backend Architecture — Part A: Initialization, Rate Limiting, Language Detection

### 5.1 Application Initialization and Lifespan

The FastAPI application is initialised using the `@asynccontextmanager` lifespan pattern:

```python
APP_VERSION = "3.2"
DEFAULT_SENTIMENT_FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_LOCAL_MODERNBERT_PATH = os.path.join(
    os.path.dirname(__file__), "models", "modernbert-sentiment"
)

@asynccontextmanager
async def lifespan(app):
    load_model()
    yield

app = FastAPI(
    title="Smart Comment Classification API",
    version=APP_VERSION,
    lifespan=lifespan
)
```

**Why `@asynccontextmanager` over `@app.on_event("startup")`?**

The `on_event("startup")` pattern was deprecated in FastAPI 0.93 in favour of the lifespan context manager. The lifespan approach:

1. Uses standard Python context manager protocol (`yield`).
2. Makes startup/shutdown logic co-located and easy to reason about.
3. Integrates with testing — `TestClient` respects the lifespan context.
4. Allows sharing of resources (e.g., database connections) between startup and shutdown phases via closure.

**Model loading at startup**: `load_model()` is synchronous and blocks the event loop during model loading. This is intentional — the server should not accept any requests before all required models are ready. FastAPI + Uvicorn will not open the port for connections until the lifespan context manager's setup phase (before `yield`) completes.

**`APP_VERSION`**: A string constant exposed in `/health` and logged at startup. Incremented on breaking API changes.

**`DEFAULT_LOCAL_MODERNBERT_PATH`**: Computed relative to `main.py`'s directory using `os.path.dirname(__file__)`. This makes the path resolution correct regardless of the working directory from which the server is launched.

### 5.2 Rate Limiting Subsystem

SCC implements an in-memory sliding-window rate limiter:

```python
rate_limit_store: dict = {}
RATE_LIMIT = 60

def check_rate_limit(client_ip: str):
    now = time.time()
    if client_ip not in rate_limit_store:
        rate_limit_store[client_ip] = []
    # Evict timestamps older than 60 seconds
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip] if now - t < 60
    ]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 60 requests per minute."
        )
    rate_limit_store[client_ip].append(now)
```

**Sliding window algorithm analysis:**

The sliding window counter is one of three standard rate limiting algorithms:

| Algorithm | Accuracy | Memory | Complexity |
|---|---|---|---|
| Fixed window | Low (burst at boundary) | O(1) per IP | Simple |
| Sliding window log | Exact | O(LIMIT) per IP | Medium |
| Sliding window counter | Approximate | O(1) per IP | Simple |

SCC uses the **sliding window log** approach (storing each timestamp individually). At RATE_LIMIT=60, memory per IP is at most 60 timestamps × 8 bytes = 480 bytes. For 1,000 concurrent unique IPs, this is ~480KB — negligible.

**Limitations of the current implementation:**

1. **No background cleanup**: Old IP entries accumulate in `rate_limit_store` indefinitely. An IP that made 60 requests once will have 60 timestamps that persist until that IP makes another request. For millions of unique IPs (bot traffic), this would grow the dict without bound. A periodic cleanup task would be needed.

2. **Process-local**: The store exists only within the Uvicorn process. Multiple Uvicorn workers (multiple processes) would each have separate stores, effectively multiplying the rate limit by the worker count. Redis-based rate limiting (e.g., using `redis-py` with `EXPIRE`) is the standard solution for distributed rate limiting.

3. **IP-based**: Behind a NAT or load balancer, many users may share the same public IP. In corporate networks, entire departments might appear as one IP. More sophisticated rate limiting would use API keys or JWT-based user identification.

4. **Not applied to file uploads**: `POST /classify/file` does not call `check_rate_limit()`. A single large file (5,000 rows) takes the server offline for other users while processing. A per-IP file upload rate limit (e.g., 5 uploads/minute) would be appropriate for production.

### 5.3 Language Detection Engine

```python
COMMON_EN_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    # ... ~100 total high-frequency English function words
}

def detect_language_is_english(text: str) -> bool:
    words = re.findall(r'[a-zA-Z]+', text.lower())
    if not words:
        return True   # Numbers/emojis only — language-neutral
    if len(words) <= 2:
        return True   # Too short to detect reliably
    en_count = sum(1 for w in words if w in COMMON_EN_WORDS)
    ratio = en_count / len(words)
    return ratio >= 0.15
```

**Design rationale — why 0.15?**

The threshold of 15% common English words is deliberately low. Consider the following informal English examples:

- `"omg lol this so bad"` — words: [omg, lol, this, so, bad], English common: [this, so] = 2/5 = 40%. Would pass a 0.15 threshold easily.
- `"smh worst update ever nothing works anymore"` — [smh, worst, update, ever, nothing, works, anymore] — English common: [nothing] = 1/7 = 14%. Borderline — slang-heavy.
- `"yaar ye product bahut acha hai"` (Hindi Romanised) — [yaar, ye, product, bahut, acha, hai] — common English: [product] = 1/6 = 16%. Would incorrectly pass as English.

The false-positive risk (Hindi Romanised detected as English) is an acknowledged limitation. A more accurate approach would use a dedicated language identification model like `fasttext` LID model (which classifies 176 languages). However, the added dependency, startup latency, and 900KB model file were considered disproportionate for a flag that only affects a UI warning, not the actual classification.

**COMMON_EN_WORDS selection**: The set contains exclusively high-frequency English **function words** (determiners, prepositions, pronouns, conjunctions, auxiliary verbs). Function words have several properties that make them ideal for language detection:

1. They appear in almost every English sentence regardless of topic (domain-invariant).
2. They are rarely borrowed into other languages verbatim (language-specific).
3. A short text that contains even 2 function words (e.g., "the" and "is") is very likely English.

**Impact on classification**: `is_english = False` is:
- Surfaced as a "Non-English text" flag pill in the UI.
- Recorded in the API response.
- It does **not** stop classification — the models still run. This design choice acknowledges that multilingual informal text (code-mixing) is common and the models can still extract partial signal.

### 5.4 Spell Correction Dictionary

The spell correction subsystem (`COMMON_TYPOS` dict + `apply_spell_correction()`) handles three categories of informal text normalisation:

**Category 1 — Standard typographic errors:**

```python
"teh": "the", "hte": "the", "taht": "that", "recieve": "receive",
"definately": "definitely", "seperate": "separate", "occured": "occurred",
"untill": "until", "wierd": "weird", "alot": "a lot",
```

These are the most common English misspellings, typically caused by finger transpositions or phonetic spelling errors. They are corrected before the transformer tokeniser sees the text, ensuring the model's subword vocabulary handles these words correctly.

**Category 2 — Missing apostrophe contractions:**

```python
"doesnt": "does not", "didnt": "did not", "isnt": "is not",
"wasnt": "was not", "cant": "cannot", "wont": "will not",
"dont": "do not", "ive": "i have", "youre": "you are",
"theyre": "they are", "theres": "there is",
```

Contractions without apostrophes (`"doesnt"`, `"cant"`) are handled here rather than in the CONTRACTIONS dict because the CONTRACTIONS dict key format requires the apostrophe (e.g., `"don't"`). The spell correction step handles the apostrophe-free variants, expanding them to full forms that the model handles well.

**Category 3 — SMS and internet shorthand:**

```python
"ppl": "people", "msg": "message", "prob": "probably", "rly": "really",
"sry": "sorry", "diff": "different", "govt": "government",
"pics": "pictures", "fav": "favorite", "gr8": "great",
"h8": "hate", "l8r": "later", "2day": "today",
"b4": "before", "dat": "that", "dis": "this",
"da": "the", "dey": "they", "dem": "them", "wat": "what",
```

This includes abbreviated forms (`ppl`, `msg`), number-substitution shorthand (`gr8=great`, `h8=hate`, `l8r=later`), and phonetic spelling from AAVE/informal dialects (`dat=that`, `dis=this`, `dey=they`).

**`apply_spell_correction()` implementation:**

```python
def apply_spell_correction(text: str) -> str:
    words = text.split()
    corrected = []
    for word in words:
        lower = word.lower().strip(".,!?;:'\"")
        if lower in COMMON_TYPOS:
            corrected.append(COMMON_TYPOS[lower])
        else:
            corrected.append(word)
    return ' '.join(corrected)
```

Key notes:
- Strips common punctuation from the word before lookup but does not reattach it — this is acceptable since the preprocessing pipeline later normalises whitespace and punctuation.
- Replacement is lowercase from the dictionary. This means `"WON'T"` would not be caught (all-caps). However, the SLANG_MAP handles all-caps common words like `"OMG"`.
- The word is split by spaces only (no subword splitting). This means compound misspellings like `"definatelynotsure"` would not be caught — only whitespace-separated words.

---

## 5. Backend Architecture — Part B: Text Preprocessing Pipeline

### 5.5 Text Preprocessing Pipeline

The preprocessing pipeline is a 11-stage function that transforms raw informal text into a cleaned form suitable for transformer tokenisation:

```python
def preprocess_text(text: str) -> str:
    if not text or not text.strip():
        return text

    # Stage 1: HTML entity decode
    text = html.unescape(text)

    # Stage 2: Unicode NFKC normalisation
    text = unicodedata.normalize("NFKC", text)

    # Stage 3: Emoji demojise
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Stage 4: URL removal
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Stage 5: @mention removal
    text = re.sub(r'@\w+', '', text)

    # Stage 6: Hashtag segmentation
    def segment_hashtag(match):
        tag = match.group(1)
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
        words = words.replace('_', ' ')
        return words
    text = re.sub(r'#(\w+)', segment_hashtag, text)

    # Stage 7: Character deduplication (3+ same chars → 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Stage 8: Punctuation deduplication (3+ → 2)
    text = re.sub(r'([!?.]){3,}', r'\1\1', text)

    # Stage 9: Spell correction
    text = apply_spell_correction(text)

    # Stage 10: Contraction and slang expansion
    words = text.split()
    expanded = []
    for word in words:
        lower = word.lower().strip(".,!?;:")
        if lower in CONTRACTIONS:
            expanded.append(CONTRACTIONS[lower])
        elif lower in SLANG_MAP and len(word) <= 10:
            expanded.append(SLANG_MAP[lower])
        else:
            expanded.append(word)
    text = ' '.join(expanded)

    # Stage 11: Whitespace normalisation
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

**Stage-by-stage transformation examples:**

| Stage | Input | Output |
|---|---|---|
| 1 HTML unescape | `"great &amp; fast"` | `"great & fast"` |
| 2 NFKC normalize | `"ＨｅｌｌＯ"` (fullwidth) | `"HellO"` |
| 3 Emoji demojize | `"great  app"` | `"great  :fire: app"` |
| 4 URL remove | `"check https://t.co/abc out"` | `"check  out"` |
| 5 @mention remove | `"@JohnDoe this is wrong"` | `" this is wrong"` |
| 6 Hashtag segment | `"#BadExperience"` | `"Bad Experience"` |
| 7 Char dedup | `"noooooo way"` | `"noo way"` |
| 8 Punct dedup | `"really???"` | `"really??"` |
| 9 Spell correct | `"gr8 awsome app"` | `"great awesome app"` |
| 10 Contractions | `"can't believe it"` | `"cannot believe it"` |
| 10 Slang | `"smh tbh this sucks"` | `"shaking my head to be honest this sucks"` |
| 11 Whitespace | `"hello   world "` | `"hello world"` |

**The `CONTRACTIONS` dictionary (70+ entries):**

The contractions map covers:
- Standard English contractions: `"ain't"`, `"aren't"`, `"can't"`, `"couldn't"`, `"didn't"`, `"doesn't"`, `"don't"` — all 31 common contractions.
- AAVE/informal contractions: `"gonna"` → `"going to"`, `"gotta"` → `"got to"`, `"wanna"` → `"want to"`, `"kinda"` → `"kind of"`, `"sorta"` → `"sort of"`, `"dunno"` → `"do not know"`, `"lemme"` → `"let me"`, `"gimme"` → `"give me"`.
- Modern informal: `"ima"` → `"i am going to"`, `"tryna"` → `"trying to"`, `"finna"` → `"fixing to"`, `"boutta"` → `"about to"`.

**The `SLANG_MAP` dictionary (50+ entries):**

The slang map covers internet acronyms and modern informal vocabulary:

```python
"brb": "be right back", "btw": "by the way",
"smh": "shaking my head", "imo": "in my opinion",
"imho": "in my humble opinion", "tbh": "to be honest",
"idk": "i do not know", "omg": "oh my god",
"lol": "laughing out loud", "lmao": "laughing my ass off",
"rofl": "rolling on the floor laughing", "ngl": "not going to lie",
"iirc": "if i recall correctly", "fwiw": "for what it is worth",
"afaik": "as far as i know", "ftw": "for the win",
"wtf": "what the fuck", "stfu": "shut the fuck up",
"nvm": "never mind", "ikr": "i know right",
"rn": "right now", "af": "as fuck",
"lowkey": "subtly", "highkey": "very much",
"srsly": "seriously", "pls": "please",
"plz": "please", "thx": "thanks",
"ty": "thank you", "np": "no problem",
"yw": "you are welcome", "ofc": "of course",
"obvi": "obviously", "tho": "though",
"nah": "no", "yep": "yes", "yup": "yes", "nope": "no",
"cuz": "because", "bc": "because",
"ur": "your", "2": "to", "4": "for", "m8": "mate",
"w/": "with", "w/o": "without", "w/e": "whatever",
```

**The `len(word) <= 10` guard on slang expansion:**

Without this guard, single-letter abbreviations like `"w"` (→ `"with"`) or `"b"` (→ `"be"`) would replace parts of longer words if the loop encountered them. The guard ensures only short standalone slang tokens are expanded. Note that `"w/"` and `"w/o"` are handled as multi-character entries, so they are caught before the length guard applies.

### 5.6 Gibberish and Nonsense Detection

Gibberish detection is one of SCC's most sophisticated heuristic systems. It prevents the ML pipeline from wasting inference time and producing meaningless output on keyboard mashing or random character strings:

```python
def is_gibberish(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    # --- Numeric gibberish check ---
    digits_only = re.sub(r'[\s\-_.,:;!?]', '', stripped)
    if digits_only.isdigit() and len(digits_only) >= 4:
        if len(digits_only) >= 8:
            return True
        if re.search(r'(\d{2,4})\1{1,}', digits_only):
            return True

    # --- Alpha content extraction ---
    alpha_only = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
    if not alpha_only:
        return False

    words = alpha_only.split()
    if not words or len(alpha_only.replace(' ', '')) <= 2:
        return False

    # --- Known-short whitelist ---
    KNOWN_SHORT = {
        'ok', 'no', 'yes', 'lol', 'omg', 'wtf', 'bruh', 'nah', 'yep',
        'hmm', 'huh', 'meh', 'ugh', 'wow', 'gg', 'ez', 'oof',
        # ... 30 more entries
    }
    if all(w in KNOWN_SHORT for w in words):
        return False

    # --- Per-word gibberish scoring ---
    gibberish_words = 0
    for word in words:
        if word in KNOWN_SHORT or len(word) <= 2:
            continue

        # Check 1: Vowel ratio
        vowels = sum(1 for c in word if c in 'aeiou')
        ratio = vowels / len(word)
        if ratio < 0.1 or ratio > 0.75:
            gibberish_words += 1
            continue

        # Check 2: Long consonant cluster (5+ consecutive)
        if re.search(r'[^aeiou]{5,}', word):
            gibberish_words += 1
            continue

        # Check 3: Character repetition pattern
        if len(word) >= 6 and re.search(r'(.{2,3})\1{2,}', word):
            gibberish_words += 1
            continue

        # Check 4: Keyboard row sequences
        keyboard_rows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        is_keyboard_seq = False
        for row in keyboard_rows:
            for start in range(len(row) - 3):
                seq = row[start:start + 4]
                if seq in word or seq[::-1] in word:
                    is_keyboard_seq = True
                    break
        if is_keyboard_seq and len(word) >= 5:
            gibberish_words += 1

    # --- Final verdict ---
    substantive_words = [w for w in words if w not in KNOWN_SHORT and len(w) > 2]
    if not substantive_words:
        return False
    return gibberish_words / len(substantive_words) >= 0.5
```

**Detailed analysis of each check:**

**Numeric gibberish**: Phone numbers (`"9876543210"`), random digit strings (`"1234567890"`), and other long pure-digit sequences are flagged. The repeating digit pattern check (`r'(\d{2,4})\1{1,}'`) catches formatted number sequences like `"1212"` or `"123123"`.

**Vowel ratio** (0.1 threshold, 0.75 threshold):
- Average English word vowel ratio: ~35% (e.g., `"beautiful"` = 5/9 = 56%, `"strength"` = 1/8 = 12.5%).
- A word with < 10% vowels is almost certainly a consonant cluster (`"xzklt"`, `"bcdfg"`).
- A word with > 75% vowels is likely random vowel sequence (`"aeouia"`, `"ouuei"`).
- Real English exceptions: `"rhythms"` (1/7 = 14.3%) would barely pass. `"strengths"` (1/9 = 11%) would be flagged — acceptable false positive for a rare word.

**Long consonant cluster**: `r'[^aeiou]{5,}'` catches strings like `"qwrtp"` or `"xzxzx"`. Most real English words do not have 5+ consecutive consonants. Exceptions: `"strengths"` has "ngths" (5 consonants), `"eighths"` has "ghths" (5 consonants). These are extremely rare.

**Character repetition**: `r'(.{2,3})\1{2,}'` catches patterns where 2-3 character sequences repeat 3+ times: `"ababab"`, `"abcabcabc"`, `"xoxoxo"`. This catches a specific pattern of keyboard mashing.

**Keyboard row sequences**: Looks for 4-character forward or backward substrings of QWERTY keyboard rows:
- `"qwert"` → contains `"qwer"` (row 1, start 0).
- `"tyuio"` → contains `"tyui"` (row 1, start 4).
- `"lkjh"` → contains `"kjh"` reversed — `"hjk"` in `"asdfghjkl"` at position 4, reversed is `"kjh"`, so `"lkjh"` would be caught.

The `len(word) >= 5` guard prevents short 4-letter keyboard sequences (which might be valid words like `"quit"` containing `"quit"` ← actually not a keyboard sequence) from being flagged.

**The `KNOWN_SHORT` whitelist:**

Short words that pass the vowel ratio check but are clearly valid informal English:
- Internet acknowledgements: `"ok"`, `"nah"`, `"yep"`, `"nope"`, `"yup"`.
- Reaction tokens: `"ugh"`, `"meh"`, `"hmm"`, `"huh"`, `"wow"`, `"oof"`.
- Gaming slang: `"gg"` (good game), `"ez"` (easy), `"fr"` (for real).
- Common abbreviations: `"brb"`, `"smh"`, `"tbh"`, `"idk"`, `"ngl"`.
- Single grammatical tokens: `"a"`, `"i"`, `"the"`, `"is"`, `"to"`, `"be"`, `"or"`, `"so"`.

A text consisting entirely of `KNOWN_SHORT` words is not gibberish — it is valid informal English.

**Short-circuit response for gibberish:**

When `is_gibberish()` returns `True`, the text record is modified before any model inference:

```python
if record["gibberish"]:
    record["predicted_sentiment"] = "Neutral"
    record["sent_scores"] = {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0}
    record["predicted_type"] = "Spam"
    record["type_scores"] = {label: (1.0 if label == "Spam" else 0.0)
                              for label in TYPE_LABEL_MAP.values()}
    record["is_uncertain"] = False
    record["emotions"] = []
    record["is_sarcastic"] = False
    record["sarcasm_score"] = 0.0
    record["heuristics_applied"].append("gibberish_short_circuit")
```

This short-circuit happens at the **response assembly stage** (after all models have already run). The model inference cannot be skipped for individual items in a batch call — it would require separating gibberish texts from the batch, which adds complexity. The short-circuit overrides results after inference, which is acceptable since the ML inference results for gibberish text are meaningless anyway.

### 5.7 Multi-Sentence Splitter

```python
def split_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) >= 3]
```

**Regex explanation:**

`r'(?<=[.!?])\s+'` uses a **lookbehind assertion** `(?<=[.!?])` to match whitespace that is immediately preceded by a sentence-ending punctuation mark. The lookbehind is zero-width — it asserts a condition without consuming characters.

Split examples:
- `"Good design. Bad battery."` → `["Good design.", "Bad battery."]`
- `"OMG!!! This is great!"` → `["OMG!!", "This is great!"]` (after punctuation dedup)
- `"Wait, what? How did that happen?"` → `["Wait, what?", "How did that happen?"]`

**Known failure cases:**

1. **Abbreviations with periods**: `"Dr. Smith is great."` → `["Dr.", "Smith is great."]` — splits on `"Dr."`. This is a known limitation of regex-based sentence splitting.
2. **Decimal numbers**: `"costs $5.99 and ships fast"` → `["costs $5.", "99 and ships fast"]` — splits on the decimal point. However, the character deduplication in preprocessing keeps decimal numbers intact, and this pattern requires the decimal to be followed by a space.
3. **Ellipsis**: `"It was ok... not great."` → after dedup: `"It was ok.. not great."` → no split (two dots are not in `[.!?]`... wait, two dots would be split). This would split on the second dot. An edge case acceptable for informal text.

The minimum fragment length of 3 characters filters out single-punctuation fragments that may result from splitting.

---

## 5. Backend Architecture — Part C: Model Loading and Registry

### 5.8 Model Loading and Registry System

SCC maintains a centrally-managed model registry that tracks each model's load status, enables graceful fallback, and exposes telemetry via the health endpoint:

**Global state:**

```python
sentiment_classifier = None
toxicity_classifier = None
type_classifier = None
emotion_classifier = None
sarcasm_classifier = None
vader_analyzer = SentimentIntensityAnalyzer()
model_tokenizers: dict = {}     # model_name -> tokenizer
model_registry: dict = {}       # model_name -> pipeline
model_status: dict = {}         # model_name -> status dict

PIPELINE_DEVICE = 0 if torch.cuda.is_available() else -1
```

**`MODEL_SPECS` dictionary:**

```python
MODEL_SPECS = {
    "sentiment": {
        "task": "sentiment-analysis",
        "display_name": "Sentiment Model",
        "top_k": None,
        "max_tokens": 512,
        "required": True,
    },
    "toxicity": {
        "task": "text-classification",
        "model": "unitary/toxic-bert",
        "display_name": "Toxic-BERT",
        "top_k": None,
        "max_tokens": 512,
        "required": True,
    },
    "type": {
        "task": "zero-shot-classification",
        "model": "facebook/bart-large-mnli",
        "display_name": "BART MNLI",
        "max_tokens": 1024,
        "required": True,
    },
    "emotion": {
        "task": "text-classification",
        "model": "SamLowe/roberta-base-go_emotions",
        "display_name": "GoEmotions",
        "top_k": 5,
        "max_tokens": 512,
        "required": False,
    },
    "sarcasm": {
        "task": "text-classification",
        "model": "cardiffnlp/twitter-roberta-base-irony",
        "display_name": "Twitter RoBERTa Irony",
        "top_k": None,
        "max_tokens": 512,
        "required": False,
    },
}
```

The `required` flag differentiates core pipeline models (failure = server degraded) from enrichment models (failure = graceful degradation with empty defaults).

**`top_k` parameter**: When set to an integer, the pipeline returns the top K labels. `top_k=5` for emotion returns the 5 most likely emotions. `top_k=None` returns all labels (used for sentiment, toxicity, sarcasm where we want the full distribution).

**`_resolve_model_candidates(name)` — Candidate chain:**

```python
def _resolve_model_candidates(name: str) -> list[dict]:
    if name == "sentiment":
        modernbert_model = _get_configured_modernbert_model()
        return [
            {
                "name": "modernbert",
                "model": modernbert_model,
                "display_name": "ModernBERT Sentiment",
                "enabled": bool(modernbert_model),
            },
            {
                "name": "twitter_roberta",
                "model": DEFAULT_SENTIMENT_FALLBACK_MODEL,
                "display_name": "Twitter RoBERTa Sentiment",
                "enabled": True,
            },
        ]
    # For all other models: single candidate from MODEL_SPECS
    return [{"name": name, "model": spec["model"], "display_name": ..., "enabled": True}]
```

Only the "sentiment" slot has multiple candidates. All other models (toxicity, type, emotion, sarcasm) have exactly one candidate — if they fail to load, that model slot remains `None`.

**`_load_pipeline(name)` — Loading with fallback:**

```python
def _load_pipeline(name: str):
    spec = MODEL_SPECS[name]
    attempted = []
    last_error = None

    for candidate in _resolve_model_candidates(name):
        if not candidate.get("enabled"):
            continue
        attempted.append(candidate["model"])
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                candidate["model"], use_fast=True
            )
            kwargs = {
                "model": candidate["model"],
                "tokenizer": tokenizer,
                "device": PIPELINE_DEVICE,
            }
            if "top_k" in spec:
                kwargs["top_k"] = spec["top_k"]
            classifier = pipeline(spec["task"], **kwargs)
            model_tokenizers[name] = tokenizer
            model_registry[name] = classifier
            _set_model_availability(name, classifier,
                actual_model=candidate["model"],
                display_name=candidate["display_name"],
                attempted_models=attempted)
            return classifier
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Candidate failed for %s: %s — %s",
                           name, candidate["model"], exc)

    raise RuntimeError(
        f"Unable to load {name} from candidates {attempted}: {last_error}"
    )
```

The `tokenizer` is loaded separately from the pipeline (even though `pipeline()` would load it internally) to store it in `model_tokenizers[name]` for the truncation utility.

**`load_model()` — Top-level orchestrator:**

```python
def load_model():
    global sentiment_classifier, toxicity_classifier, type_classifier
    global emotion_classifier, sarcasm_classifier

    LOGGER.info("Loading NLP model stack on device=%s", PIPELINE_DEVICE)

    for model_name in MODEL_SPECS:
        try:
            _load_pipeline(model_name)
        except Exception as exc:
            LOGGER.exception("Failed to load %s", model_name)
            model_registry[model_name] = None
            _set_model_availability(model_name, None, str(exc), ...)

    sentiment_classifier = model_registry.get("sentiment")
    toxicity_classifier = model_registry.get("toxicity")
    # ... etc

    missing_required = [
        name for name, status in model_status.items()
        if status["required"] and not status["loaded"]
    ]
    if missing_required:
        LOGGER.error("Required models unavailable: %s", ", ".join(missing_required))
```

Loading order follows `MODEL_SPECS` insertion order (Python 3.7+ dicts maintain insertion order): sentiment → toxicity → type → emotion → sarcasm. This order ensures the most critical model (sentiment) loads first and any VRAM pressure is known early.

**`_set_model_availability()` — Status recording:**

```python
def _set_model_availability(name, classifier, error=None,
                            actual_model=None, display_name=None,
                            attempted_models=None):
    spec = MODEL_SPECS[name]
    model_status[name] = {
        "loaded": classifier is not None,
        "required": spec["required"],
        "model": actual_model or _get_primary_model_identifier(name),
        "display_name": display_name or spec.get("display_name"),
        "max_tokens": spec["max_tokens"],
        "error": error,
        "attempted_models": attempted_models or [],
    }
```

This status dict is serialised directly into the `/health` response, giving operators full visibility into which model is active and what was tried.

### 5.9 Tokenizer-Aware Truncation

```python
def truncate_for_model(text: str, model_name: str) -> tuple[str, dict]:
    tokenizer = model_tokenizers.get(model_name)
    spec = MODEL_SPECS[model_name]

    if not text or tokenizer is None:
        return text, {"truncated": False, "max_tokens": spec["max_tokens"],
                      "input_tokens": 0}

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=spec["max_tokens"],
        return_overflowing_tokens=True,
        return_attention_mask=False,
    )

    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    overflow = encoded.get("overflowing_tokens") or []
    truncated = bool(overflow)

    prepared = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    return prepared, {
        "truncated": truncated,
        "max_tokens": spec["max_tokens"],
        "input_tokens": len(token_ids),
    }
```

**The critical importance of tokenizer-aware truncation:**

Transformer models have hard token limits (typically 512 or 1024). A text exceeding this limit must be truncated. The naive approach — truncate at N characters — is problematic because:

1. Different characters tokenise to different numbers of tokens. A character like `"é"` might tokenise to 2 tokens (`"é"` → `"Ã"` + `"©"` in byte-level BPE). A 512-character truncation could produce 600+ tokens.
2. Cutting in the middle of a BPE subword unit produces a malformed token that maps to `[UNK]`.
3. The tokeniser adds special tokens (`[CLS]`, `[SEP]` for BERT models, `<s>`, `</s>` for RoBERTa). These consume positions from the max_length budget.

**How tokenizer-aware truncation works:**

1. `tokenizer(text, truncation=True, max_length=512)` tokenises the full text, then truncates to exactly 512 tokens including special tokens.
2. `return_overflowing_tokens=True` captures the tokens that were dropped, allowing detection of whether truncation occurred.
3. `tokenizer.decode(token_ids, skip_special_tokens=True)` converts the (possibly truncated) token IDs back to a clean text string.
4. The decoded text is what is passed to the pipeline, ensuring the model never sees more tokens than its position embedding allows.

**`return_attention_mask=False`**: Attention masks are not needed at the truncation step — they will be generated by the pipeline internally when it tokenises the prepared text.

**Per-model truncation metadata** in the response:

```json
"truncation": {
    "sentiment": { "truncated": false, "max_tokens": 512, "input_tokens": 24 },
    "toxicity":  { "truncated": false, "max_tokens": 512, "input_tokens": 24 },
    "type":      { "truncated": false, "max_tokens": 1024, "input_tokens": 24 },
    "emotion":   { "truncated": false, "max_tokens": 512, "input_tokens": 24 },
    "sarcasm":   { "truncated": false, "max_tokens": 512, "input_tokens": 24 }
}
```

Each model has its own truncation entry because different models have different max_tokens and different tokenisers (RoBERTa BPE vs BART BPE) which may tokenise the same text to different lengths.

### 5.10 Batch Inference Engine

```python
def _run_stage_batch(model_name: str, texts: list[str], **kwargs) -> tuple[list, float]:
    classifier = model_registry.get(model_name)
    if classifier is None:
        return [None] * len(texts), 0.0

    started = time.perf_counter()
    results = classifier(texts, batch_size=min(16, max(1, len(texts))), **kwargs)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    if len(texts) == 1:
        if model_name == "type":
            if isinstance(results, dict):
                results = [results]
        elif isinstance(results, dict):
            results = [results]
        elif isinstance(results, list) and results and isinstance(results[0], dict):
            results = [results]
    elif not isinstance(results, list):
        results = [results]

    return results, elapsed_ms
```

**Why `batch_size=min(16, max(1, len(texts)))`:**

The `batch_size` parameter tells HuggingFace pipelines how many texts to process simultaneously in a single GPU matrix multiplication. GPU efficiency is highest when the batch is large enough to saturate the GPU's parallel processing units. The constraints:

- `max(1, len(texts))` ensures at least 1 (prevents `batch_size=0` which would be invalid).
- `min(16, ...)` caps at 16 items per GPU batch. This is a conservative cap chosen to:
  - Prevent VRAM overflow for long texts (16 × 512 tokens × 768 hidden dims is manageable on most GPUs).
  - Keep latency predictable — a very large batch (e.g., 5,000 items) would take minutes.
  - Allow the progress counter in the job store to update frequently.

**Result normalisation for single vs. batch:**

HuggingFace pipelines have inconsistent output formats for single vs. batch inputs:
- Single input to sentiment pipeline: `[{"label": "positive", "score": 0.87}, ...]` (list of dicts)
- Single input to zero-shot classification: `{"labels": [...], "scores": [...]}` (single dict)
- Batch input (n>1): Always a list of results.

The normalisation code handles these inconsistencies:
- For type model (zero-shot): wraps single dict in `[...]`.
- For other models: if result is a list of dicts (not list of lists), wraps in outer `[...]`.
- Result is always `list[result_for_item]` for downstream processing.

**Stage timings:**

`time.perf_counter()` provides the highest-resolution timer available on the platform (nanosecond precision on most systems). The `elapsed_ms` value is stored in `record["stage_timings_ms"][model_name]` and returned in the API response, enabling the development team to profile which model is the bottleneck.

---

*End of Part 1 — Continues in Part 2*
