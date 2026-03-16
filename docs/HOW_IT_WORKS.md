# How It Works — Technical Deep Dive

A full breakdown of every stage in the Smart Comment Classification pipeline, each model used, and the reasoning behind design decisions.

---

## Pipeline Overview

```
User Input
    │
    ▼
[1] Text Preprocessing
    │  → emoji conversion, URL removal, hashtag segmentation
    │  → spell correction, slang/contraction expansion
    │  → repeated character normalization
    ▼
[2] Gibberish Detection
    │  → keyboard mash? numeric spam? → early exit (Neutral / Spam)
    ▼
[3] Language Detection
    │  → is it English? (flags non-English, still processes)
    ▼
[4] Sentiment Analysis          [5] Sarcasm Detection
    │  twitter-roberta               twitter-roberta-irony
    │                                    │
    └──────── sarcasm flip? ────────────┘
    ▼
[6] Toxicity Detection          [7] Emotion Detection
    │  toxic-bert                    go_emotions (28 emotions)
    ▼
[8] Comment Type Classification
    │  bart-large-mnli (zero-shot)
    │  + heuristic boosting
    │  + emotion-informed boosting
    │  + sarcasm-type correction
    ▼
[9] Confidence Thresholding
    │  → < 55% confidence? flag as uncertain
    ▼
[10] Multi-Sentence Aggregation
     │  → 2+ sentences? classify each, detect mixed sentiment
     ▼
[11] Word-Level Analysis (VADER)
     │  → per-word sentiment with context window
     ▼
API Response → Frontend renders results
```

---

## Stage 1 — Text Preprocessing

Before any model sees the input, the text goes through a normalization pipeline to handle informal English.

| Step | What it does | Example |
|---|---|---|
| HTML unescaping | Decode HTML entities | `&amp;` → `&` |
| Unicode normalization | Normalize character variants | `ﬁ` → `fi` |
| Emoji conversion | Convert emoji to text descriptions | `😂` → `:face with tears of joy:` |
| URL removal | Strip links | `https://...` → `` |
| Mention removal | Strip @handles | `@user` → `` |
| Hashtag segmentation | Split camelCase hashtags | `#LoveThis` → `Love This` |
| Repeated char normalization | Collapse excessive repetition | `loooove` → `loove` |
| Spell correction | Fix ~150 common typos | `teh` → `the`, `ur` → `your` |
| Contraction expansion | Expand contracted forms | `can't` → `cannot` |
| Slang expansion | Expand internet slang | `smh` → `shaking my head` |

**Why preprocess?** Neural models tokenize text into subword pieces. "loooove" gets split into weird tokens the model rarely saw during training. After normalization it becomes "loove" or "love" — tokens the model understands well. This is the single biggest accuracy improvement for informal text.

---

## Stage 2 — Gibberish Detection

Random keyboard mashing is detected and short-circuited before hitting any expensive model.

| Check | Catches |
|---|---|
| Vowel ratio < 10% | `sdfjklsdf`, `zxcvbnm` |
| Consonant clusters > 5 chars | `sdqwdfgh` |
| Keyboard row sequences | `qwertyuiop`, `asdfghjkl` |
| Repeated substring patterns | `abcabcabc` |
| Long pure numeric strings (8+ digits) | `123123123123` |
| Repeated digit patterns | `12121212` |

When gibberish is detected: sentiment → Neutral (100%), type → Spam, all other fields cleared.

---

## Stage 3 — Language Detection

A lightweight word-overlap check against a set of ~100 common English words. If fewer than 15% of alphabetic words match, the comment is flagged as non-English with a blue badge. The pipeline still runs — the models handle many languages reasonably — but the flag alerts users that results may be less reliable.

---

## The Models

### Model 1 — Sentiment Analysis
**`cardiffnlp/twitter-roberta-base-sentiment-latest`**

- **Architecture:** RoBERTa-base (125M parameters), fine-tuned on ~124 million tweets
- **What it does:** Classifies the overall emotional tone as **Positive**, **Neutral**, or **Negative**
- **Why this model:** Trained on Twitter data, so it natively understands informal language — abbreviations like "smh", "lmao", "ngl", mixed casing, missing punctuation, and internet slang. A model trained on formal text badly misclassifies "ngl this slaps no cap" (should be Positive).
- **Output:** Three probability scores (positive / neutral / negative) summing to 1.0
- **Shown as:** The main sentiment badge + the three confidence bars at the bottom of the result card

---

### Model 2 — Sarcasm / Irony Detection
**`cardiffnlp/twitter-roberta-base-irony`**

- **Architecture:** RoBERTa-base (125M parameters), fine-tuned on the SemEval 2018 Task 3 irony detection dataset (Twitter)
- **What it does:** Detects whether a comment is using irony — saying the opposite of what's literally meant
- **Why this matters:** Without it, "Oh great, only crashed 5 times today!" scores as Positive (model sees "great"). With sarcasm detection, sentiment flips to Negative.
- **False positive prevention:** Two guards prevent over-triggering on genuinely enthusiastic text:
  1. **Negative cues required:** Sarcasm only flagged if model scores > 80% AND text contains words like crash, broken, fail, terrible, etc.
  2. **Gratitude suppression:** If text contains thank-you language (thanks, tysm, appreciate), the negative cue check is bypassed — genuine gratitude is never flagged as sarcasm
- **Downstream effects:** If sarcastic + sentiment was Positive → flip to Negative. If sarcastic + type was Praise → redistribute score toward Complaint.
- **Shown as:** Purple "Sarcasm detected (X%)" flag badge

---

### Model 3 — Toxicity Detection
**`unitary/toxic-bert`**

- **Architecture:** BERT-base (110M parameters), fine-tuned on the Jigsaw Toxic Comment Classification dataset (~160K Wikipedia comments labeled by human raters)
- **What it does:** Scores how abusive, hateful, or harmful a comment is (0.0–1.0)
- **Why this model:** The Jigsaw dataset covers threats, insults, obscenity, and identity-based hate — purpose-built for community moderation. It's far more reliable for toxicity than general sentiment models.
- **Threshold:** > 0.5 → flagged as toxic
- **Downstream effect:** Toxic + Negative sentiment → boosts Complaint type score
- **Shown as:** Red "Toxic" badge + red progress bar showing the percentage

---

### Model 4 — Emotion Detection
**`SamLowe/roberta-base-go_emotions`**

- **Architecture:** RoBERTa-base (125M parameters), fine-tuned on Google's GoEmotions dataset (~58K Reddit comments across 28 emotion categories)
- **What it does:** Identifies specific emotions present in the comment
- **The 28 emotions:** admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral
- **Why emotions?** Sentiment alone gives you positive/neutral/negative. Emotions tell you *why* — anger vs disappointment are both negative but suggest completely different responses. Emotions also improve type classification accuracy.
- **Emotion → Type mapping:**

  | Emotion | Boosts Type |
  |---|---|
  | anger, annoyance, disappointment, disgust | Complaint |
  | admiration, gratitude, joy, love, approval, excitement | Praise |
  | confusion, curiosity | Question |
  | desire, optimism | Feedback |

- **Threshold:** Only emotions with score > 0.30 influence type boosting
- **Shown as:** Emotion tags with percentages in the result card

---

### Model 5 — Comment Type Classification (Zero-Shot)
**`facebook/bart-large-mnli`**

- **Architecture:** BART-large (400M parameters), fine-tuned on Multi-Genre Natural Language Inference (MNLI). Used in zero-shot mode via the NLI pipeline.
- **What it does:** Classifies comments into one of six types without any task-specific training:
  - **Praise** — appreciation, compliments, thanks
  - **Complaint** — dissatisfaction, frustration, criticism
  - **Question** — asking for help or information
  - **Feedback** — constructive suggestions, feature requests
  - **Spam** — promotional, off-topic, or irrelevant content
  - **Other** — general observations
- **How zero-shot NLI works:** The model treats each candidate type as a hypothesis — e.g. "This text is expressing praise or appreciation." It scores how strongly the input *entails* that hypothesis. The highest-scoring hypothesis wins. No labeled training data for comment types is needed.
- **Why BART-large:** It's one of the strongest zero-shot classifiers available and generalizes well to custom label sets.
- **Heuristic boosting layer** on top of the raw model scores:

  | Signal | Effect |
  |---|---|
  | `?` or question words at start | Boost Question |
  | Slang praise words (`fire`, `bussin`, `goat`, `no cap`) | Boost Praise ×1.35, suppress Complaint ×0.4 |
  | Superlative complaints (`worst ever`, `most useless`) | Boost Complaint +0.4 |
  | Spam keywords + URLs + all-caps + excess `!` | Boost Spam multiplicatively |
  | Feedback language (`suggest`, `should`, `would be nice`) | Boost Feedback |
  | Emotion signals from Model 4 | Fine-tune based on emotion-type mapping |
  | Sarcasm confirmed | Redirect Praise → Complaint |

- **Shown as:** The type badge (Question, Praise, Complaint, etc.)

---

### Model 6 — Word-Level Analysis
**VADER (Valence Aware Dictionary and sEntiment Reasoner)**

- **Architecture:** Rule-based lexicon, not a neural model. ~7,500 words and phrases with pre-assigned valence scores, plus grammatical rules for negation, capitalization, punctuation, and booster words.
- **What it does:** Scores every individual word in the original (unpreprocessed) text as Positive, Neutral, or Negative
- **Why VADER instead of a neural model?** Neural models classify the full text as a unit — they can't tell you which specific word was positive or negative. VADER operates at word level, making it ideal for the highlighted word visualization. It's also instant (no GPU) and reliable for lexical sentiment.
- **Context window:** Each word is scored in context of the 2 preceding words. This handles negation — "not good" scores differently because "not" is in the window.
- **Shown as:** The color-highlighted text display (green = positive, red = negative, grey = neutral) and word count stats

---

## Confidence Thresholding

If the top sentiment score is below **55%**, the result is flagged with a yellow "Low confidence" badge. This happens with:
- Genuinely ambiguous or mixed-signal text
- Very short inputs
- Unusual phrasing the models haven't seen much

---

## Multi-Sentence Aggregation

For comments with 2+ sentences, each sentence is classified separately by the sentiment model:

1. Each sentence gets its own Positive / Neutral / Negative scores
2. If sentences disagree → flagged as **mixed sentiment**
3. Final sentiment = average of per-sentence scores
4. Per-sentence breakdown shown in the result card with colored dots

**Example:**
```
"The food was great but the service was terrible. I loved the dessert though."

Sentence 1: Negative (89%)   ●
Sentence 2: Positive (97%)   ●
Overall:    Mixed → averaged → Uncertain
```

---

## Why These Models Were Chosen

The original version used ModernBERT-base for sentiment and a lighter zero-shot model for types. Problems:
- ModernBERT was not trained on informal/social text → poor slang handling
- The lighter NLI model was less accurate on the 6-type classification task
- No sarcasm, emotion, or multi-sentence handling at all

Replacements:
- `twitter-roberta` for sentiment → trained on 124M tweets, informal text native
- `bart-large-mnli` for types → stronger NLI backbone, better zero-shot generalization
- Added `go_emotions` → richer signal for type classification and user-facing insight
- Added `twitter-roberta-irony` → catches the most common source of false-positive sentiment

---

## Accuracy Progression

| Version | What changed | Sentiment | Type | Combined |
|---|---|---|---|---|
| v1.0 | ModernBERT + basic NLI | ~70% | ~60% | ~65% |
| v2.0 | twitter-roberta + bart-large + heuristics | 76% | 80% | 78% |
| v3.0 | + emotions + sarcasm + spell correction + gibberish detection | 84% | 92% | **88%** |
