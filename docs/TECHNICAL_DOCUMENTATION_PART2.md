# Smart Comment Classification (SCC)
## Complete Technical Documentation — Part 2 of 3

**Continued from Part 1**

---

## Part 2 Contents

- Section 5 (Parts E–H) — Backend Architecture:
  - Sentiment Normalization
  - Sarcasm-Aware Adjustment
  - Type Classification with Heuristics
  - Confidence Thresholding
  - Multi-Sentence Aggregation
  - Word-Level VADER Analysis
  - Job Queue and Background Processing
  - API Endpoints
- Section 6 — ML Models In-Depth Analysis
- Section 7 — ModernBERT Fine-Tuning Pipeline

---

## 5. Backend Architecture — Part E: Inference and Sarcasm

### 5.11 Sentiment Normalization

Raw sentiment model output varies by model architecture. Twitter RoBERTa labels its outputs `LABEL_0`, `LABEL_1`, `LABEL_2`, while the fine-tuned ModernBERT uses `Negative`, `Neutral`, `Positive`. The normaliser bridges these differences:

```python
def _normalize_sentiment_output(result) -> dict:
    scores = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    items = (result[0] if isinstance(result, list)
                       and result
                       and isinstance(result[0], list)
             else result)

    for item in items:
        label = item["label"].lower()
        if "positive" in label:
            scores["Positive"] += item["score"]
        elif "negative" in label:
            scores["Negative"] += item["score"]
        else:
            scores["Neutral"] += item["score"]

    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 4) for k, v in scores.items()}
    return scores
```

**Label matching strategy:**

The normaliser uses substring matching on the lowercased label:
- `"LABEL_2"` → `"label_2"` → does not contain "positive", "negative" → goes to Neutral bucket.

Wait — this is a problem. `"LABEL_0"`, `"LABEL_1"`, `"LABEL_2"` do not contain the substrings `"positive"` or `"negative"`. For Twitter RoBERTa, the `top_k=None` setting returns ALL labels including their scores. The label→sentiment mapping for Twitter RoBERTa is embedded in the model's `config.json`:

```json
{
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  }
}
```

The HuggingFace pipeline for Twitter RoBERTa actually returns `"Negative"`, `"Neutral"`, `"Positive"` — the label names as defined in the model's fine-tuning config. More precisely, the Cardiff NLP models use the label names `"negative"`, `"neutral"`, `"positive"` in their config (all lowercase). The normaliser's `"positive" in label.lower()` correctly catches `"positive"`, `"Positive"`, and any model that includes "positive" as a substring.

**Re-normalisation after matching:**

Multiple model labels might be summed into the same bucket if a model uses label names like `"very_positive"` and `"slightly_positive"` — the sum would be correctly captured. The `/ total` re-normalisation handles cases where floating-point accumulation causes the sum to not be exactly 1.0.

**Result: Always a clean `{ "Positive": float, "Neutral": float, "Negative": float }` dict** — downstream code never needs to handle raw model label names.

### 5.12 Sarcasm-Aware Sentiment Adjustment

Sarcasm detection is one of SCC's most nuanced features. The sarcasm model detects ironic language, but not all ironic language means the expressed sentiment should be flipped. The `_apply_sarcasm_adjustment` function implements a context-gated sarcasm flip:

```python
def _apply_sarcasm_adjustment(record: dict, predicted_sentiment: str,
                               sent_scores: dict, sarcasm_score: float):
    lower_text = record["text"].lower()

    # Gate 1: Does the text contain negative situation cues?
    has_negative_cues = bool(re.search(
        r'\b(crash|broke|broken|fail|problem|issue|wrong|bad|worst|terrible|'
        r'horrible|sucks|hate)\b|(\d+\s*times)',
        lower_text
    ))

    # Gate 2: Is this actually a genuine expression of gratitude?
    has_gratitude = bool(re.search(
        r'\b(thanks?|thx|ty|tysm|thank you|grateful|appreciate|'
        r'thats? great|good job|nice work|well done|awesome work)\b',
        lower_text
    ))

    # Gratitude overrides negative cues
    if has_gratitude:
        has_negative_cues = False
        record["heuristics_applied"].append("sarcasm_gratitude_suppression")

    # Both conditions must hold for sarcasm flip
    is_sarcastic = sarcasm_score > 0.80 and has_negative_cues

    if is_sarcastic:
        record["heuristics_applied"].append("sarcasm_negative_context")

    # Apply score inversion only for Positive → Negative flip
    if is_sarcastic and predicted_sentiment == "Positive":
        record["heuristics_applied"].append("sarcasm_flip_positive_to_negative")
        predicted_sentiment = "Negative"

        old_pos = sent_scores.get("Positive", 0)
        old_neg = sent_scores.get("Negative", 0)
        sent_scores["Positive"] = round(old_neg * 0.5, 4)
        sent_scores["Negative"] = round(old_pos * 0.8, 4)
        sent_scores["Neutral"] = round(
            max(0.0, 1.0 - sent_scores["Positive"] - sent_scores["Negative"]), 4
        )

    return predicted_sentiment, sent_scores, is_sarcastic
```

**Why three gates are needed:**

**Gate 1 — Irony model confidence > 0.80:**

The Cardiff NLP Twitter RoBERTa Irony model was trained on the SemEval-2018 Task 3 dataset, which contains ~4,600 tweets labelled for irony. This is a relatively small training set, so the model has high uncertainty. Setting the threshold at 0.80 means we only act on high-confidence irony predictions, reducing false positives.

**Gate 2 — Explicit negative context words:**

A sarcastic remark requires two components: (1) a positively-valenced surface form, and (2) a negatively-valenced implied meaning. The negative cue words (`"crash"`, `"broke"`, `"terrible"`, `"sucks"`) identify the negative situation that is being commented on sarcastically. Without Gate 2, the following would be incorrectly flipped:

- `"Oh great, you're absolutely wonderful!"` — genuinely positive enthusiastic praise. Gate 2 fails (no negative cues), so no flip despite high irony score.

The `\d+\s*times` pattern catches repetition-of-failure patterns: `"crashed 5 times"`, `"failed 3 times"` — these indicate the negative situation without using explicit negative vocabulary.

**Gate 3 — Gratitude suppression:**

`"Thanks for the quick response!"` might score high on the irony detector if the reviewer had mixed feelings (or if the model is uncertain). The gratitude detection regex catches genuine expressions of thanks and overrides the negative cue flag, preventing an inappropriate sarcasm flip. This handles the common customer service interaction style:

- `"Great, the app crashed AGAIN. Thanks for nothing."` — irony_score = 0.87, has_negative_cues = True (crashed, nothing), has_gratitude = True (thanks). Gratitude suppression fires → is_sarcastic = False. This is actually wrong — "thanks for nothing" is sarcastic gratitude. However, the model's irony score for this text would likely be very high, and the current implementation would suppress the flip due to "thanks". This is an acknowledged limitation.

**Score inversion formula:**

When the flip occurs:
- New Negative = old Positive × 0.80 (80% of the original positive confidence becomes negative)
- New Positive = old Negative × 0.50 (50% of the original negative confidence becomes positive residual)
- New Neutral = max(0, 1 - new_Positive - new_Negative)

Example: original scores `{Positive: 0.82, Neutral: 0.12, Negative: 0.06}`:
- New Negative = 0.82 × 0.80 = 0.656
- New Positive = 0.06 × 0.50 = 0.030
- New Neutral = 1.0 - 0.030 - 0.656 = 0.314

The attenuated inversion (×0.80 rather than ×1.0) reflects that sarcasm detection itself carries uncertainty — a 0.82 irony score means ~18% chance this is NOT sarcastic. The new scores reflect this residual uncertainty.

---

## 5. Backend Architecture — Part F: Type Heuristics and Confidence

### 5.13 Type Classification with Heuristic Boosting

**Zero-shot foundation:**

BART MNLI evaluates each candidate label as an NLI hypothesis:
- Premise: `"This product is amazing, I love it."`
- Hypothesis 1: `"This is an example of expressing praise or appreciation or compliment."` → Entailment probability: 0.84
- Hypothesis 2: `"This is an example of making a complaint or expressing dissatisfaction."` → Entailment probability: 0.03
- etc.

The verbose label descriptions are carefully designed to be semantically distinctive and to contain the key discriminative vocabulary (e.g., "complaint" and "dissatisfaction" for Complaint type).

**`apply_type_heuristics()` — Complete Rule Catalog:**

```python
def apply_type_heuristics(text: str, type_scores: dict,
                           sentiment: str, tox_score: float,
                           emotions: list = None) -> dict:
    lower = text.lower().strip()
    boosted = dict(type_scores)
    heuristic_flags = []
    ...
    return boosted, heuristic_flags
```

The function modifies the `boosted` dict in-place and collects applied rule names in `heuristic_flags`.

**Rule 1 — Emotion-based boosting:**

```python
EMOTION_TYPE_BOOST = {
    "anger":           ("Complaint", 0.15),
    "annoyance":       ("Complaint", 0.12),
    "disappointment":  ("Complaint", 0.10),
    "disgust":         ("Complaint", 0.12),
    "admiration":      ("Praise",    0.15),
    "gratitude":       ("Praise",    0.18),
    "joy":             ("Praise",    0.10),
    "love":            ("Praise",    0.12),
    "approval":        ("Praise",    0.10),
    "excitement":      ("Praise",    0.10),
    "confusion":       ("Question",  0.12),
    "curiosity":       ("Question",  0.15),
    "desire":          ("Feedback",  0.08),
    "optimism":        ("Feedback",  0.06),
}
```

For each detected emotion with score > 0.3, the corresponding type receives a boost of `boost_val × emotion_score`. A `"gratitude"` emotion at 0.90 confidence boosts `Praise` by `0.18 × 0.90 = 0.162`. This creates a virtuous cycle where a text with strong gratitude emotion is pushed toward the Praise type category.

The 0.3 emotion score threshold prevents very weak emotion predictions from polluting the type scores.

**Rule 2 — Question detection:**

```python
question_patterns = [
    r'\?',
    r'^(what|where|when|why|how|who|which|is|are|was|were|do|does|did|can|could|would|should|will|shall|has|have|had)\b',
    r'^(anybody|anyone|anything|somebody|someone|something)\b',
    r'\b(how come|what if|is it|isn\'t it|aren\'t they)\b',
]
question_signal = sum(1 for p in question_patterns if re.search(p, lower))
if question_signal >= 1:
    boosted["Question"] = boosted.get("Question", 0) + 0.25 * question_signal
```

Each matched question pattern contributes 0.25 to the Question score boost. Multiple patterns matching multiply the boost:
- A text with `?` AND starting with `"how"` gets `0.25 × 2 = 0.50` boost.
- A text with `?` AND `"how come"` gets `0.25 × 2 = 0.50` boost.
- The maximum boost from question detection is `0.25 × 4 = 1.0` (if all 4 patterns match), which would dominate after renormalisation.

**Rule 3 — Uncertainty language:**

```python
uncertainty_patterns = r'\b(idk|i do not know|not sure|unsure|uncertain|maybe|might|on the fence|hard to say|can\'t decide|cannot decide|i guess|idk if|not sure if)\b'
if re.search(uncertainty_patterns, lower):
    boosted["Other"] += 0.15
    boosted["Complaint"] *= 0.5
```

Uncertainty patterns indicate the commenter is ambivalent — they don't definitively like or dislike the product. This boosts "Other" (general comment) and halves the Complaint score. A Complaint requires conviction; uncertainty contradicts that.

**Rule 4 — Complaint keyword boost:**

```python
complaint_words = r'\b(terrible|horrible|awful|worst|hate|sucks|broken|unusable|unacceptable|ridiculous|disgusting|pathetic|waste|scam|fraud|rip.?off|garbage|trash|useless|disappointed|frustrating|infuriating)\b'
if re.search(complaint_words, lower):
    boosted["Complaint"] += 0.20
```

Direct negative vocabulary that strongly predicts complaints. The regex uses `rip.?off` to match both `"rip-off"` and `"ripoff"`.

**Rule 5 — Sentiment + Toxicity double-signal:**

```python
if sentiment == "Negative" and tox_score > 0.3:
    boosted["Complaint"] += 0.15
```

A text that is both Negative in sentiment AND toxic in content is very likely a complaint (angry customer, abusive feedback). The 0.3 toxicity threshold is lower than the is_toxic flag (0.5) to catch borderline cases.

**Rule 6 — Superlative complaint:**

```python
superlative_complaint = r'\b(stupidest|dumbest|worst|most annoying|most useless|most pathetic|most ridiculous)\b.*\b(ever|in my life|of all time|i have ever|i\'ve ever)\b'
if re.search(superlative_complaint, lower):
    boosted["Complaint"] += 0.40
```

Superlative complaint constructions like `"worst app I've ever used"` or `"most useless feature of all time"` are strong complaint signals. The +0.40 is double the standard keyword boost, reflecting the high precision of this pattern.

**Rule 7 — Praise keywords:**

```python
praise_words = r'\b(amazing|awesome|excellent|fantastic|wonderful|brilliant|outstanding|perfect|incredible|love|loved|loving|best|great|superb|magnificent|phenomenal|thank|thanks|grateful|impressed)\b'
if re.search(praise_words, lower):
    boosted["Praise"] += 0.20
if sentiment == "Positive":
    boosted["Praise"] += 0.10
```

Stacked: if a text contains praise keywords AND has Positive sentiment, Praise gets +0.30 total boost.

**Rule 8 — Modern slang praise detection:**

```python
slang_praise_words = r'\b(fire|goat|lit|sick|bussin|bussing|slaps|slap|dope|heat|chef\'s kiss|elite|insane|valid|hard|goes hard|hits different|peak|god.?tier|next level|top.?tier|no cap|on point|clean|mint|ace|bangin|banging|killer|banger)\b'
if re.search(slang_praise_words, lower):
    boosted["Praise"] += 0.35
    boosted["Complaint"] *= 0.4
```

Gen Z and internet slang terms for high praise. `"This product is fire"` or `"This app bussin ngl"` are clear praise statements that the BART MNLI model (trained on formal NLI text) might misclassify. The +0.35 Praise boost and 60% reduction of Complaint score ensure these are correctly classified.

**Rule 9 — Feedback keywords:**

```python
feedback_words = r'\b(should|could|would be better|suggest|suggestion|recommend|consider|improve|improvement|perhaps|it would be nice|feature request|adding|please add|wish|hope)\b'
if re.search(feedback_words, lower):
    boosted["Feedback"] += 0.20
```

Constructive language patterns that indicate the user is providing suggestions for improvement rather than expressing pure praise or complaint.

**Rules 10-14 — Spam signals (multiplicative):**

The spam detection uses multiplicative score boosting rather than additive boosting. This reflects the compound nature of spam — a text with URLs, ALL CAPS, multiple exclamation marks, AND spam keywords is far more likely spam than a text with just one of these signals:

```python
spam_multiplier = 1.0

# Keyword matches (up to 3x accumulation)
spam_keyword_matches = len(re.findall(spam_keywords, lower))
if spam_keyword_matches >= 1:
    spam_multiplier *= (1.0 + 0.4 * spam_keyword_matches)

# URL presence (strong signal)
if re.search(r'(www\.|\.com|\.net|\.org|\.io|https?://)', lower):
    spam_multiplier *= 1.8

# Multiple exclamation marks
exclamation_count = len(re.findall(r'!', text))
if exclamation_count >= 2:
    spam_multiplier *= (1.0 + 0.2 * min(exclamation_count, 5))

# ALL CAPS words
caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
if len(caps_words) >= 1:
    spam_multiplier *= (1.0 + 0.3 * min(len(caps_words), 5))

# Imperative commands
if re.search(imperative_cmds, lower):
    spam_multiplier *= 1.5

# Apply multiplier to base spam score
if spam_multiplier > 1.0:
    base_spam = max(boosted.get("Spam", 0), 0.08)
    boosted["Spam"] = base_spam * spam_multiplier
    if spam_multiplier >= 2.5:
        boosted["Praise"] *= 0.3
```

A text with all 5 spam signals: `spam_multiplier = (1 + 0.4) × 1.8 × (1 + 0.2×5) × (1 + 0.3×5) × 1.5 = 1.4 × 1.8 × 2.0 × 2.5 × 1.5 = 18.9`. With `base_spam = 0.08`, the boosted spam score would be `0.08 × 18.9 = 1.512`. After normalisation (dividing all scores by their sum), Spam would dominate.

**Post-heuristic normalisation:**

```python
total = sum(boosted.values())
if total > 0:
    boosted = {k: round(v / total, 4) for k, v in boosted.items()}
```

Re-normalisation converts the boosted scores back to a valid probability distribution summing to 1.0.

**Final sarcasm-type correction:**

```python
if record["is_sarcastic"] and record["predicted_sentiment"] == "Negative":
    praise_val = type_scores.get("Praise", 0)
    type_scores["Complaint"] += praise_val * 0.6
    type_scores["Praise"] = praise_val * 0.3
    total_t = sum(type_scores.values())
    if total_t > 0:
        type_scores = {k: round(v / total_t, 4) for k, v in type_scores.items()}
    record["heuristics_applied"].append("sarcasm_type_redirect")
```

If sarcasm was detected and the sentiment was flipped to Negative, the type should reflect this. The BART model may have assigned high Praise probability (because the surface text was positive), but after sarcasm detection we know the comment is actually negative. 60% of the Praise score is transferred to Complaint, and Praise is reduced to 30%.

### 5.14 Confidence Thresholding

```python
CONFIDENCE_THRESHOLD = 0.55

def apply_confidence_flag(sentiment: str, sent_scores: dict) -> tuple:
    max_conf = max(sent_scores.values())
    is_uncertain = max_conf < CONFIDENCE_THRESHOLD
    return sentiment, is_uncertain
```

**Statistical rationale for 0.55 threshold:**

In a 3-class balanced classification problem:
- **Random baseline**: 33.3% confidence per class.
- **Slight preference**: 40% confidence — barely above chance.
- **Moderate confidence**: 55% confidence — meaningful signal, but notable uncertainty.
- **High confidence**: 75%+ confidence — strong prediction.

The 0.55 threshold marks the boundary between "the model has some meaningful signal" and "the model is essentially guessing". Below this boundary, displaying the predicted label might mislead the user — the "Low confidence" flag communicates this appropriately.

**When is confidence low?**

1. **Mixed-sentiment texts**: `"I love the design but hate the performance."` Both Positive and Negative get ~45% each, leaving no dominant class above 0.55.
2. **Short texts**: `"ok"`, `"not bad"` — insufficient signal for strong classification.
3. **Ambiguous framing**: `"It could be better."` — neither clearly Negative (complaint) nor clearly Neutral (factual observation).
4. **After sarcasm inversion**: The score inversion in `_apply_sarcasm_adjustment` spreads confidence across classes, sometimes dropping all below 0.55.

**Uncertainty in batch results**: In the `BatchResults` component, `is_uncertain` rows are tagged with a "Low confidence" pill badge in the comments table. The KPI card counts the total number of uncertain rows for the entire batch.

---

## 5. Backend Architecture — Part G: Multi-Sentence and VADER

### 5.15 Multi-Sentence Aggregation

```python
def classify_multi_sentence(text: str, cleaned: str) -> dict:
    sentences = split_sentences(cleaned)

    if len(sentences) <= 1:
        return None

    per_sentence = []
    agg_scores = {"Positive": 0, "Neutral": 0, "Negative": 0}

    for sent_text in sentences:
        try:
            sentence_input, _ = truncate_for_model(sent_text, "sentiment")
            result = sentiment_classifier(sentence_input)
            scores = _normalize_sentiment_output(result)
            predicted = max(scores, key=scores.get)
            per_sentence.append({
                "text": sent_text,
                "sentiment": predicted,
                "scores": scores
            })
            for k in agg_scores:
                agg_scores[k] += scores.get(k, 0)
        except Exception:
            per_sentence.append({
                "text": sent_text,
                "sentiment": "Neutral",
                "scores": {"Positive": 0, "Neutral": 1, "Negative": 0}
            })
            agg_scores["Neutral"] += 1

    n = len(per_sentence)
    if n > 0:
        agg_scores = {k: round(v / n, 4) for k, v in agg_scores.items()}

    return {
        "sentences": per_sentence,
        "aggregated_scores": agg_scores,
        "aggregated_sentiment": max(agg_scores, key=agg_scores.get),
        "is_mixed": len(set(s["sentiment"] for s in per_sentence)) > 1
    }
```

**When multi-sentence aggregation applies:**

The function is called for all texts but only acts when `split_sentences()` returns 2 or more sentences. For single-sentence texts, it returns `None` and the primary classification scores are used.

**Aggregation logic:**

Each sentence is classified independently by the sentiment model. Scores are summed and divided by count (simple average). The `is_mixed` flag is `True` when at least two sentences have different predicted labels — meaning the text contains mixed sentiment.

**Effect on primary scores:**

When `is_mixed = True`, the aggregated scores replace the single-pass scores:

```python
if multi_sentence and multi_sentence["is_mixed"]:
    record["sent_scores"] = multi_sentence["aggregated_scores"]
    record["predicted_sentiment"] = multi_sentence["aggregated_sentiment"]
    record["predicted_sentiment"], record["is_uncertain"] = apply_confidence_flag(
        record["predicted_sentiment"], record["sent_scores"]
    )
    record["heuristics_applied"].append("mixed_sentiment_aggregation")
```

This means for a text like "I love the design. The battery is terrible. Overall it's ok.", the per-sentence analysis provides a more nuanced view than the single-pass model which would see the full concatenated text.

**Performance cost:**

Multi-sentence analysis runs an additional `N-1` model inference calls (where N is the sentence count). For a 5-sentence text, this means 4 additional sentiment classifications. On GPU this is fast (~20ms each), but on CPU this can add 400ms × 4 = 1.6 seconds to the total latency.

**API response structure for multi-sentence:**

```json
"multi_sentence": {
    "sentences": [
        { "text": "I love the design.", "sentiment": "Positive",
          "scores": { "positive": 0.91, "neutral": 0.07, "negative": 0.02 } },
        { "text": "The battery is terrible.", "sentiment": "Negative",
          "scores": { "positive": 0.04, "neutral": 0.08, "negative": 0.88 } },
        { "text": "Overall it's ok.", "sentiment": "Neutral",
          "scores": { "positive": 0.18, "neutral": 0.72, "negative": 0.10 } }
    ],
    "is_mixed": true
}
```

### 5.16 Word-Level Sentiment Analysis (VADER)

```python
def analyze_word_sentiment(text: str) -> tuple[list, dict]:
    tokens = re.findall(r'\S+|\s+', text)
    word_analysis = []
    word_counts = {"total": 0, "positive": 0, "neutral": 0, "negative": 0}
    word_positions = []
    word_to_position = {}

    for token in tokens:
        if token.strip():
            position = len(word_positions)
            word_positions.append(token)
            word_to_position.setdefault(token, []).append(position)
            word_counts["total"] += 1

    seen_counts = Counter()
    rebuilt_analysis = []

    for token in tokens:
        if not token.strip():
            rebuilt_analysis.append({"text": token, "sentiment": "Whitespace"})
            continue

        occurrence = seen_counts[token]
        positions = word_to_position.get(token, [])
        word_idx = positions[occurrence] if occurrence < len(positions) else len(word_positions) - 1
        seen_counts[token] += 1

        # Context window: current word + 2 preceding words
        context_start = max(0, word_idx - 2)
        context_words = word_positions[context_start:word_idx + 1]
        context_phrase = ' '.join(context_words)

        if len(context_words) > 1:
            phrase_score = vader_analyzer.polarity_scores(context_phrase)['compound']
            single_score = vader_analyzer.polarity_scores(token)['compound']
            # Use phrase score if it disagrees with word score (negation detection)
            if (phrase_score > 0 > single_score) or (phrase_score < 0 < single_score):
                score = phrase_score
            else:
                score = single_score
        else:
            score = vader_analyzer.polarity_scores(token)['compound']

        if score >= 0.05:
            sent = "Positive"
        elif score <= -0.05:
            sent = "Negative"
        else:
            sent = "Neutral"

        rebuilt_analysis.append({"text": token, "sentiment": sent})
        word_counts[sent.lower()] += 1

    return rebuilt_analysis, word_counts
```

**Algorithm walkthrough:**

The function processes text token by token, preserving whitespace as `"Whitespace"` sentinel tokens to allow the frontend to reconstruct the original text layout with inter-word spaces.

**Context window for negation:**

For each word, VADER evaluates:
1. The word in isolation (single-word score).
2. The word in a 2-word context window preceding it (phrase score).

If the phrase score and single-word score **disagree in sign** (one is positive, the other negative), the phrase score is used. This captures the most common negation case: `"not good"` — `"good"` has a positive single score, but `"not good"` has a negative phrase score → the negative phrase score is used.

**Token-to-position mapping:**

The `word_to_position` dict maps each unique token to a list of all its positions in the text. The `seen_counts` Counter tracks how many times each token has been processed. This allows the algorithm to correctly associate the Nth occurrence of a repeated word with its actual position in the sequence (needed for the correct context window).

**Output structure:**

The `word_analysis` list has one entry per token (including whitespace):
```json
[
    { "text": "This",     "sentiment": "Neutral" },
    { "text": " ",        "sentiment": "Whitespace" },
    { "text": "app",      "sentiment": "Neutral" },
    { "text": " ",        "sentiment": "Whitespace" },
    { "text": "is",       "sentiment": "Neutral" },
    { "text": " ",        "sentiment": "Whitespace" },
    { "text": "amazing",  "sentiment": "Positive" }
]
```

The frontend iterates this list and renders each token in a `<span>` with the appropriate CSS class (`highlight-positive`, `highlight-negative`, `highlight-neutral`). Whitespace tokens render as plain `<span>` with just the space character.

---

## 5. Backend Architecture — Part H: Jobs and API

### 5.17 Job Queue and Background Processing

**In-memory job store:**

```python
jobs: dict = {}

def process_batch_job(job_id: str, texts: list):
    job = jobs[job_id]
    job["status"] = "processing"
    results = []

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_results = classify_texts_internal(
                [str(text) for text in batch]
            )
        except Exception:
            LOGGER.exception("Batch inference failed for rows %d-%d",
                             i, i + len(batch) - 1)
            batch_results = [None] * len(batch)

        for text, result in zip(batch, batch_results):
            try:
                if result is None:
                    raise RuntimeError("No result from batch inference")
                results.append({
                    "comment":       str(text),
                    "sentiment":     result["sentiment"],
                    "conf_pos":      result["sentiment_confidence"]["positive"],
                    "conf_neu":      result["sentiment_confidence"]["neutral"],
                    "conf_neg":      result["sentiment_confidence"]["negative"],
                    "toxicity":      result["toxicity"],
                    "is_toxic":      result["is_toxic"],
                    "comment_type":  result["comment_type"],
                    "is_sarcastic":  result.get("is_sarcastic", False),
                    "is_uncertain":  result.get("is_uncertain", False),
                    "emotions":      result.get("emotions", []),
                    "word_analysis": result.get("word_analysis", []),
                    "word_counts":   result.get("word_counts", {}),
                    "stage_timings_ms": result.get("stage_timings_ms", {}),
                    "heuristics_applied": result.get("heuristics_applied", []),
                    "truncation":    result.get("truncation", {}),
                })
            except Exception as e:
                results.append({
                    "comment": str(text),
                    "sentiment": "Error",
                    "conf_pos": 0, "conf_neu": 0, "conf_neg": 0,
                    "toxicity": 0, "is_toxic": False,
                    "comment_type": "Unknown",
                    "is_sarcastic": False, "is_uncertain": False,
                    "emotions": [], "word_analysis": [], "word_counts": {},
                    "stage_timings_ms": {},
                    "heuristics_applied": [f"batch_error:{type(e).__name__}"],
                    "truncation": {},
                })
            job["processed"] = len(results)

    job["status"] = "done"
    job["results"] = results
```

**Job lifecycle states:**

```
"queued" -> "processing" -> "done"
```

Jobs transition from `queued` → `processing` immediately when `process_batch_job` begins. `"queued"` is only visible if the frontend polls before the background task starts (within the first event loop tick). In practice, the background task starts within milliseconds.

**Error isolation levels:**

Level 1 — Full batch call failure: If `classify_texts_internal()` throws for an entire 16-item batch (e.g., unexpected model error), all 16 items get `[None]` placeholders. The try/except wrapping the batch call prevents one bad batch from aborting the entire job.

Level 2 — Individual item error: If result processing for a specific item within a successful batch call throws, that item gets an error sentinel `{"sentiment": "Error", ...}`. The `heuristics_applied` field contains `"batch_error:ExceptionClassName"` for diagnostic purposes.

**Progress tracking granularity:**

`job["processed"] = len(results)` is updated **after each individual item** is appended to results. This means for a 250-row file processed in batches of 16, the progress counter increments 1 at a time (not 16 at a time). This gives the frontend smooth progress updates rather than chunky 6% jumps.

### 5.18 API Endpoints — Implementation Details

**`GET /health`:**

```python
@app.get("/health")
async def health():
    required_ready = all(
        model_status.get(name, {}).get("loaded")
        for name, spec in MODEL_SPECS.items()
        if spec["required"]
    )
    optional_ready = all(
        model_status.get(name, {}).get("loaded")
        for name, spec in MODEL_SPECS.items()
        if not spec["required"]
    )
    return {
        "status": "ok" if (required_ready and optional_ready)
                       else ("degraded" if required_ready
                             else "model_not_loaded"),
        ...
    }
```

Three-way status logic:
- `"ok"` — all 5 models loaded.
- `"degraded"` — required models (sentiment, toxicity, type) loaded; optional (emotion, sarcasm) failed.
- `"model_not_loaded"` — at least one required model failed to load.

This is surfaced in the React frontend's NavBar as a status indicator chip: green (ok), yellow (degraded), red (offline/model_not_loaded).

**`POST /classify/text`:**

```python
@app.post("/classify/text")
async def classify_text_endpoint(request: Request):
    check_rate_limit(request.client.host)

    body = await request.json()
    text = body.get("text", "").strip()

    if not text:
        raise HTTPException(status_code=400,
            detail="Please enter a comment. Text field cannot be empty.")
    if len(text) > 8192:
        raise HTTPException(status_code=400,
            detail="Text exceeds maximum length of 8192 characters.")

    result = classify_text_internal(text)
    return JSONResponse(content=result)
```

The endpoint uses `request: Request` rather than a Pydantic model parameter. This is intentional — it allows the rate limiter to access `request.client.host` (client IP) before input parsing. With Pydantic body parsing, the framework would validate the body first, potentially executing the body parser before the rate limit check.

The 8192 character limit is chosen to match ModernBERT's native 8K token context. A character:token ratio of ~1:1 for English means 8192 characters ≈ 8000 tokens, which exactly fills ModernBERT's context window. For standard RoBERTa (512 token limit), long texts are truncated by `truncate_for_model()`.

**`POST /classify/file`:**

```python
@app.post("/classify/file")
async def classify_file_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    column: Optional[str] = Form(None)
):
    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in [".csv", ".txt", ".xlsx"]):
        raise HTTPException(status_code=400,
            detail="Only .csv, .txt, .xlsx files are accepted.")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400,
            detail="File exceeds maximum size of 10MB.")

    # ... parse file, extract texts ...

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "total": len(texts), "processed": 0, "results": []}
    background_tasks.add_task(process_batch_job, job_id, texts)

    return JSONResponse(content={"job_id": job_id, "status": "processing",
                                  "total_rows": len(texts)})
```

The `await file.read()` reads the entire file content into memory as bytes. This is appropriate for files up to 10MB but would need streaming (`UploadFile.read(chunk_size)`) for larger files. The 10MB limit prevents memory exhaustion.

`Optional[str] = Form(None)` — the `column` parameter is extracted from the multipart form data (not the file body). It is `None` when the form submits only the file, and a string column name on second submission.

`str(uuid.uuid4())[:8]` generates an 8-character hex string job ID. UUID4 uses 122 bits of randomness; the first 8 hex characters use 32 bits, giving a collision probability of ~1 in 4 billion per job creation. At SCC's expected scale (hundreds of jobs), this is negligible.

---

## 6. ML Models — In-Depth Analysis

### 6.1 Sentiment Model — ModernBERT and Twitter RoBERTa

**6.1.1 Twitter RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)**

Architecture: RoBERTa-base — 12 Transformer encoder layers, 768 hidden dimensions, 12 attention heads, 125M parameters.

**RoBERTa architecture details:**

RoBERTa (Robustly Optimised BERT Pre-training Approach, Liu et al. 2019) builds on BERT with four key modifications:

1. **Dynamic masking**: BERT generates the masking pattern once during data preprocessing. RoBERTa regenerates the masking pattern for each epoch, forcing the model to learn from different token combinations.

2. **Removal of NSP**: BERT's Next Sentence Prediction (NSP) objective trains the model to predict whether sentence B follows sentence A. Liu et al. found NSP harmful to downstream task performance and removed it in RoBERTa.

3. **Larger batches**: RoBERTa trains with batch size 8,192 (vs BERT's 256), using gradient accumulation to simulate larger effective batches on hardware-constrained setups.

4. **More data and longer training**: RoBERTa trains on 160GB of text (Books+Wiki+CC-News+OpenWebText+Stories) for 100K steps vs BERT's 40K steps.

**Twitter-specific fine-tuning (TweetEval):**

Cardiff NLP's Twitter RoBERTa was fine-tuned on the TweetEval benchmark (Barbieri et al. 2020), specifically the `sentiment` task. TweetEval-sentiment contains ~45K annotated tweets from SemEval 2017 Task 4, with three sentiment classes:
- Positive: tweets expressing satisfaction, approval, excitement.
- Neutral: factual or mixed-sentiment tweets.
- Negative: tweets expressing dissatisfaction, anger, sadness.

Performance on TweetEval: F1 ~0.72 (macro averaged across 3 classes).

**Why Twitter data matters for SCC:**

Standard sentiment datasets (SST-2, IMDb) contain formal English from movie reviews and academic text. Twitter data is informal, contains emoji, hashtags, abbreviations, sarcasm, code-mixing — exactly the vocabulary distribution of user comments that SCC processes.

**6.1.2 ModernBERT (answerdotai/ModernBERT-base)**

Architecture: 22 encoder layers, 768 hidden dimensions, 12 attention heads, 149M parameters.

**Attention mechanism — Alternating global/local:**

ModernBERT uses two types of attention layers:
- **Local layers** (every layer that is NOT a multiple of 3): Sliding window attention with window size 128. Each token attends only to tokens within 64 positions before and after it.
- **Global layers** (every 3rd layer, i.e., layers 3, 6, 9, 12, 15, 18, 21): Full attention where every token attends to every other token.

This design reduces the computational complexity from O(L²) to approximately O(L × 128 × 2/3 + L² × 1/3) where L is the sequence length. For a 1024-token sequence, global attention would require 1,048,576 attention computations, while local attention requires only 262,144. The periodic global layers maintain long-range dependencies for the full sequence.

**RoPE (Rotary Position Embeddings):**

In standard BERT, position information is added to token embeddings before the first attention layer. In ModernBERT, position is encoded as a rotation of the Query and Key vectors in each attention head:

```
Q' = Q * cos(m*θ) + rotate(Q) * sin(m*θ)
K' = K * cos(n*θ) + rotate(K) * sin(n*θ)
```

Where `m` and `n` are the absolute positions, and `θ` is a frequency parameter per dimension. The dot product `Q' · K'` depends only on the relative position `m - n`, making RoPE a relative position encoding scheme. This property means the model can extrapolate to sequence lengths not seen during training (unlike absolute positional embeddings which have a fixed range).

**GeGLU activation:**

Traditional transformer feed-forward layers use:
```
FFN(x) = GeLU(xW₁ + b₁)W₂ + b₂
```

GeGLU (Gated Linear Unit with GeLU gate) uses:
```
FFN(x) = (GeLU(xW₁ + b₁) ⊙ (xW₂ + b₂))W₃ + b₃
```

The gating mechanism (`⊙` is element-wise multiplication) allows the network to selectively suppress or amplify activations, giving it more expressivity per layer.

**Training corpus:**

ModernBERT was pre-trained on 2 trillion tokens from:
- Web crawl data (Common Crawl, cleaned).
- Code (GitHub repositories).
- Scientific papers (Semantic Scholar, ArXiv).

The inclusion of code and scientific text (unlike BERT which was trained only on Books+Wikipedia) gives ModernBERT better generalisation to technical vocabulary, which is beneficial for product reviews containing technical terminology.

**Fine-tuning for SCC:**

The local checkpoint at `backend/models/modernbert-sentiment/` was produced by `train_modernbert_sentiment.py` using:
- Dataset: `cardiffnlp/tweet_eval` (sentiment task)
- Training steps: 4,278 (saved at both 2,852 and 4,278)
- Labels: `Negative=0`, `Neutral=1`, `Positive=2`
- Metric: macro F1 (checkpoint at step 2,852 was intermediate; final at 4,278)

### 6.2 Toxicity Model — Toxic-BERT (unitary/toxic-bert)

**Training data — Jigsaw Toxic Comment Challenge:**

The Jigsaw Toxic Comment Classification Challenge (2018) provided ~160K Wikipedia talk page comments annotated for six toxicity categories:

| Category | Description | Prevalence |
|---|---|---|
| toxic | Contains rude or disrespectful language | 9.6% |
| severe_toxic | Extremely offensive or threatening | 1.0% |
| obscene | Uses obscene language | 5.3% |
| threat | Threatens harm | 0.3% |
| insult | Personally insulting | 4.9% |
| identity_hate | Attacks based on identity | 0.9% |

The dataset has multi-label annotations — a comment can be toxic AND obscene AND insulting simultaneously. Toxic-BERT was trained with sigmoid cross-entropy (multi-label) rather than softmax (single-label).

**Why SCC uses only the `toxic` label:**

Using just the binary `toxic` signal simplifies the user-facing output without losing much practical utility. The `toxic` label is the broadest and most commonly useful signal for content moderation purposes. Displaying all 6 categories would add complexity without commensurate value for most SCC users.

The raw score is used as a continuous measure of abusiveness, not just binary is/is-not toxic. Scores of 0.3-0.5 might indicate borderline language worth reviewing, while scores > 0.9 indicate strongly toxic content.

**Impact on Type classification:**

Toxicity interacts with Type classification in two heuristics:
1. `sentiment == "Negative" and tox_score > 0.3` → boost Complaint by 0.15 (a toxic-negative comment is likely a vented complaint).
2. `spam_multiplier >= 2.5` → the type scoring already reduces Praise substantially, indirectly accounting for spam being non-genuine.

### 6.3 Comment Type Model — BART MNLI (facebook/bart-large-mnli)

**BART architecture:**

BART (Bidirectional and Auto-Regressive Transformers, Lewis et al. 2019) is an encoder-decoder model. The encoder processes the input text bidirectionally (like BERT). The decoder generates output autoregressively (like GPT). This makes BART particularly effective for generation tasks (summarisation, translation).

For NLI (Natural Language Inference), the model uses a classification head on top of the encoder's representation of the `[EOS]` token — the same position where the decoder's generation begins. This position aggregates information from the entire input.

**MultiNLI fine-tuning:**

MultiNLI (Williams et al. 2018) contains ~433K premise-hypothesis pairs with three labels:
- Entailment: The hypothesis follows from the premise.
- Contradiction: The hypothesis contradicts the premise.
- Neutral: The hypothesis is unrelated to or not inferrable from the premise.

For zero-shot classification, only the Entailment probability is used as the "score" for each candidate label.

**Template design for comment type:**

The template `"This is an example of [label]."` is a standard NLI zero-shot template. More specific templates can improve accuracy. For example:

- `"This example is about [label]."` (more factual)
- `"The author of this text is [label]."` (agent-focused)
- `"The purpose of this text is [label]."` (intent-focused)

SCC uses the verbose descriptions (`"expressing praise or appreciation or compliment"`) rather than short labels (`"Praise"`) because BART MNLI was trained on formal NLI datasets where the hypothesis contains full English phrases. Short single-word hypotheses give the model insufficient context to make confident entailment judgements.

**BART max_tokens = 1024:**

BART-large was pre-trained with a 1024-token encoder. SCC sets `max_tokens=1024` for the type model while other models use 512. This allows longer comments to be fully processed by the type classifier even when they would be truncated for the other models.

### 6.4 Emotion Model — GoEmotions (SamLowe/roberta-base-go_emotions)

**GoEmotions dataset:**

GoEmotions (Demszky et al. 2020) was created by Google and contains ~58K Reddit comments annotated for 27 emotions plus "neutral". Annotators selected all applicable emotions from the 28-label set for each comment.

The dataset covers a broad spectrum from basic emotions (anger, fear, joy, sadness, surprise, disgust) to cognitive and social emotions (admiration, curiosity, confusion, gratitude, pride, embarrassment, remorse).

**28 emotion categories:**

```
Positive (high valence):
  admiration, amusement, approval, caring, excitement, gratitude, joy,
  love, optimism, pride, relief

Negative (low valence):
  anger, annoyance, disappointment, disapproval, disgust, embarrassment,
  fear, grief, nervousness, remorse, sadness

Ambiguous / Cognitive:
  confusion, curiosity, desire, realization, surprise

Neutral:
  neutral
```

**Multi-label nature:**

A comment can express multiple emotions simultaneously. `top_k=5` returns the 5 most probable emotions with their individual scores. These are softmax probabilities over the 28 classes, with the top 5 typically covering > 90% of the total probability mass.

**Integration with Type heuristics:**

The emotion→type boost (`EMOTION_TYPE_BOOST`) uses emotion detection to refine type classification. Key mappings:

| Emotion | Type Boosted | Boost | Rationale |
|---|---|---|---|
| gratitude (0.18) | Praise | Highest | Thank-you comments are almost always Praise |
| curiosity (0.15) | Question | High | Curious commenters ask questions |
| admiration (0.15) | Praise | High | Admiration = genuine appreciation |
| anger (0.15) | Complaint | High | Anger = complaint |
| confusion (0.12) | Question | Medium | Confused commenters seek clarification |
| annoyance (0.12) | Complaint | Medium | Annoyed customers complain |
| disgust (0.12) | Complaint | Medium | Disgust = strong dissatisfaction |
| love (0.12) | Praise | Medium | Love = strong positive sentiment |

**Threshold**: Only emotions with score > 0.30 trigger boosts. Weak emotions (score = 0.05-0.15) are common background noise in the GoEmotions model and should not influence type classification.

### 6.5 Sarcasm/Irony Model — Twitter RoBERTa Irony

**SemEval 2018 Task 3:**

The training data came from SemEval 2018 Task 3 (Hee et al. 2018) — Irony Detection in English Tweets. The dataset contains ~3,834 tweets labelled for:
- Subtask A: Ironic vs. Non-ironic (binary).
- Subtask B: Verbal irony (contrast), Situational irony (unexpected outcome), Other irony types, Non-ironic.

Cardiff NLP's model was trained on Subtask A (binary irony detection). The two classes are `non_irony` (LABEL_0) and `irony` (LABEL_1).

**SCC label extraction:**

```python
sarcasm_score = _extract_label_score(result, {"irony", "label_1", "1"})
```

The label name for irony may be `"irony"`, `"label_1"`, or `"1"` depending on the model configuration. The set-based lookup handles all three cases.

**Why sarcasm is classified as Optional (`required: False`):**

Sarcasm detection has a relatively high false-positive rate even at the 0.80 threshold. Making it optional means:
1. If the Cardiff NLP model fails to download (network issue), the server still works correctly without sarcasm detection.
2. The system can be tested or run in resource-constrained environments (CPU-only, low RAM) without needing the sarcasm model.

When the sarcasm model is not loaded, `sarcasm_score = 0.0` and `is_sarcastic = False` for all comments — the sentiment is never flipped. This is a conservative safe default.

### 6.6 VADER Lexicon — Word-Level Analysis

**VADER construction methodology:**

The VADER lexicon was constructed using a 5-step process:

1. **Seed collection**: Gather known positive and negative sentiment features (words, acronyms, emoticons, punctuation).
2. **Wisdom-of-the-crowd rating**: Use Amazon Mechanical Turk to get human ratings (-4 to +4) for each feature.
3. **Statistical validation**: Compute inter-rater agreement (Jensen-Shannon divergence). Only features with high agreement are included.
4. **Rule incorporation**: Add hand-crafted rules for punctuation (!, ALL CAPS), boosters, negation.
5. **Validation**: Test on 4 different social media corpora (Twitter, Yelp, Amazon, NY Times).

**VADER in SCC — specific usage pattern:**

VADER is used exclusively for the word-level token highlighting feature. It is NOT used for the primary sentiment classification (that comes from the transformer models). This division of labour is deliberate:

| Attribute | Transformer Models | VADER |
|---|---|---|
| Accuracy | High | Moderate |
| Speed | Slow (100ms+) | Instant (<1ms) |
| Context awareness | Full sentence | 2-word window |
| Informal text handling | Excellent | Good |
| Use in SCC | Primary sentiment | Per-word highlight |

---

## 7. ModernBERT Fine-Tuning Pipeline

### 7.1 Overview and Purpose

`backend/training/train_modernbert_sentiment.py` is a complete, self-contained fine-tuning script that adapts `answerdotai/ModernBERT-base` to the 3-class sentiment classification task. It is designed to:

1. Accept training data from either a local file (CSV/XLSX/TXT) or a Hugging Face dataset.
2. Automatically split training data into train/validation sets if no separate evaluation file is provided.
3. Fine-tune ModernBERT with optimal hyperparameters.
4. Save the trained checkpoint to the local `backend/models/modernbert-sentiment/` directory.
5. Generate a metadata JSON file for audit purposes.

### 7.2 Command-Line Interface

The script uses Python's `argparse` module with comprehensive options:

```bash
# Option A: Local CSV file
python backend/training/train_modernbert_sentiment.py \
    --train-file data/comments.csv \
    --eval-file data/eval_comments.csv \
    --text-column comment \
    --label-column label_true \
    --epochs 3 \
    --learning-rate 2e-5 \
    --train-batch-size 8 \
    --output-dir backend/models/my-model

# Option B: HuggingFace dataset
python backend/training/train_modernbert_sentiment.py \
    --dataset-name cardiffnlp/tweet_eval \
    --dataset-config sentiment \
    --text-column text \
    --label-column label \
    --epochs 5 \
    --gradient-accumulation-steps 4

# Option C: Local data with auto train/val split
python backend/training/train_modernbert_sentiment.py \
    --train-file data/comments.csv \
    --text-column comment \
    --label-column label \
    --validation-split 0.15 \
    --seed 42
```

**Arguments reference:**

| Argument | Default | Description |
|---|---|---|
| `--train-file` | None | Path to CSV/XLSX/TXT training file |
| `--eval-file` | None | Evaluation file (omit = auto-split from train) |
| `--dataset-name` | None | HuggingFace dataset ID |
| `--dataset-config` | None | Dataset config/subset (e.g., "sentiment") |
| `--train-split` | "train" | Split name in HF dataset for training |
| `--eval-split` | None | Split name for eval; auto-detects "validation"/"test" |
| `--text-column` | "comment" | Column containing text |
| `--label-column` | "label_true" | Column containing sentiment labels |
| `--base-model` | "answerdotai/ModernBERT-base" | Base checkpoint to fine-tune |
| `--output-dir` | backend/models/modernbert-sentiment | Save directory |
| `--max-length` | 512 | Maximum token length |
| `--epochs` | 3 | Number of training epochs |
| `--learning-rate` | 2e-5 | Peak learning rate |
| `--train-batch-size` | 8 | Per-device train batch size |
| `--eval-batch-size` | 8 | Per-device eval batch size |
| `--weight-decay` | 0.01 | L2 regularisation coefficient |
| `--gradient-accumulation-steps` | 2 | Steps before weight update |
| `--validation-split` | 0.2 | Fraction of train data for validation |
| `--seed` | 42 | Random seed for reproducibility |

### 7.3 Data Loading Pipeline

**Local file loading:**

```python
def load_table(path: str) -> pd.DataFrame:
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(src)
    if suffix == ".xlsx":
        return pd.read_excel(src, engine="openpyxl")
    if suffix == ".txt":
        return pd.DataFrame({
            "comment": [
                line.strip()
                for line in src.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        })
    raise ValueError(f"Unsupported data file: {src}")
```

TXT format wraps each non-empty line in a DataFrame with a single column named `"comment"`. This allows TXT training files to be used with `--text-column comment`.

**Label normalisation:**

```python
LABEL_TO_ID = {"Negative": 0, "Neutral": 1, "Positive": 2}

def normalize_label(label: Any) -> str:
    cleaned = str(label).strip().capitalize()
    if cleaned not in LABEL_TO_ID:
        raise ValueError(f"Unsupported label '{label}'.")
    return cleaned
```

`str(label).strip().capitalize()` converts `"positive"` → `"Positive"`, `"NEGATIVE"` → `"Negative"`, `3` → `"3"` (which would raise ValueError). This ensures consistent label handling regardless of the original capitalisation in the training file.

**HuggingFace integer label mapping:**

```python
TWEET_EVAL_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

def build_examples_from_hf(records, text_column, label_column):
    for row in records:
        raw_label = row[label_column]
        if isinstance(raw_label, (int, np.integer)) and int(raw_label) in TWEET_EVAL_LABELS:
            label = TWEET_EVAL_LABELS[int(raw_label)]
        else:
            label = normalize_label(raw_label)
```

TweetEval stores labels as integers (0, 1, 2). The `TWEET_EVAL_LABELS` map converts these to the SCC label names. The `isinstance(raw_label, (int, np.integer))` check handles both Python `int` and NumPy integer types.

**Evaluation split strategy:**

```python
def load_examples(args):
    if args.dataset_name:
        train_examples = build_examples_from_hf(dataset[args.train_split], ...)
        if args.eval_split:
            eval_examples = build_examples_from_hf(dataset[args.eval_split], ...)
        elif "validation" in dataset:
            eval_examples = build_examples_from_hf(dataset["validation"], ...)
        elif "test" in dataset:
            eval_examples = build_examples_from_hf(dataset["test"], ...)
        else:
            train_examples, eval_examples = split_examples(train_examples, ...)
```

The priority order for evaluation data:
1. Explicit `--eval-split` argument.
2. `"validation"` split in the HF dataset (if present).
3. `"test"` split in the HF dataset (if present and no validation).
4. Random split of training data.

TweetEval has both train and validation splits, so `dataset["validation"]` is used by default.

### 7.4 Dataset Class and Tokenisation

```python
@dataclass
class EncodedSentimentDataset(Dataset):
    encodings: dict[str, Any]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index])
                for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def make_dataset(tokenizer, examples, max_length):
    encodings = tokenizer(
        [example["text"] for example in examples],
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    labels = [example["label"] for example in examples]
    return EncodedSentimentDataset(encodings=encodings, labels=labels)
```

**`@dataclass` with PyTorch `Dataset`:**

Using `@dataclass` reduces boilerplate — the `encodings` and `labels` fields are automatically set by the dataclass `__init__`. The class still satisfies the PyTorch `Dataset` interface by implementing `__len__` and `__getitem__`.

**Pre-tokenisation:**

All examples are tokenised in a single batch call (`tokenizer([text1, text2, ...], ...)`). The resulting `encodings` dict contains:
- `input_ids`: 2D list [n_examples × max_length] of token IDs.
- `attention_mask`: 2D list [n_examples × max_length] of 0/1 values.
- (For BERT tokenisers) `token_type_ids`: 2D list of 0s (single-sequence input).

The `padding=True` parameter pads all sequences to the length of the longest sequence in the batch (or `max_length` if shorter sequences exist). The DataLoader batches these pre-padded sequences.

**On-demand tensor conversion in `__getitem__`:**

The encodings are stored as Python lists, not tensors. Conversion to `torch.tensor()` happens per-item in `__getitem__`. This is memory-efficient — the full dataset stays in CPU memory as Python lists, and only the current batch is converted to tensors (which may be moved to GPU).

### 7.5 Training Configuration and Hyperparameters

```python
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
```

**Detailed hyperparameter analysis:**

**Learning Rate (2e-5):**
The optimal learning rate for fine-tuning transformer encoders is typically in the range [1e-5, 5e-5]. Values above 1e-4 cause catastrophic forgetting (the pre-trained weights are overwritten). Values below 1e-6 result in extremely slow convergence. 2e-5 is the most commonly reported optimal value in the literature for BERT-family models on classification tasks.

**Weight Decay (0.01):**
L2 regularisation prevents overfitting by penalising large weight values. The 0.01 coefficient adds `0.01 × ||W||²` to the loss. This is applied to all weights except biases and LayerNorm parameters (the HuggingFace Trainer handles this exclusion automatically).

**Warmup Steps (10% of total steps):**
At the beginning of training, gradient magnitudes are large and noisy. A linear learning rate warmup ramps the LR from 0 → peak_LR over the warmup steps, preventing large gradient updates that could destabilise the pre-trained weights in the first epoch.

**Cosine LR Scheduler:**
After warmup, the learning rate follows a cosine schedule: `LR(t) = peak_LR × 0.5 × (1 + cos(π × t/T_max))`. The cosine schedule decays more smoothly than linear decay and typically achieves better final model performance.

**Evaluation and Save Strategy (per epoch):**
The model is evaluated on the validation set at the end of each epoch. If the macro F1 improves, the checkpoint is saved. `load_best_model_at_end=True` ensures the final model corresponds to the best checkpoint, not the last one.

**`save_total_limit=2`:**
Only the 2 most recent checkpoints are kept on disk. This prevents the output directory from filling with N_epochs checkpoint directories.

**Mixed Precision Training (fp16):**
On CUDA-capable GPUs, `fp16=True` enables FP16 (half-precision) training. Activations and gradients are stored in FP16 (16-bit floating point) while the master weights remain in FP32 (32-bit). This approximately halves GPU memory usage and accelerates computation on NVIDIA's Tensor Cores. The PyTorch `GradScaler` handles gradient scaling to prevent underflow in FP16.

**Gradient Accumulation (steps=2):**
Effective batch size = `per_device_batch_size × gradient_accumulation_steps` = 8 × 2 = 16. Gradient accumulation simulates a larger batch size without requiring more GPU memory. Gradients are accumulated over 2 forward/backward passes before the weight update, which provides more stable gradient estimates than single mini-batches of 8.

### 7.6 Custom Metric — Macro F1

```python
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
```

**Why macro F1 over accuracy:**

Sentiment datasets are typically imbalanced — Neutral comments are often more frequent than Positive or Negative. Accuracy weighted by class frequency would allow a model that predicts Neutral for everything to achieve high accuracy. Macro F1 computes F1 independently for each class and then averages, treating all classes equally regardless of frequency.

**F1 score decomposition:**

For each class `c`:
- **Precision_c** = TP_c / (TP_c + FP_c) — of all predictions labeled c, how many were actually c?
- **Recall_c** = TP_c / (TP_c + FN_c) — of all actual c examples, how many were predicted as c?
- **F1_c** = 2 × Precision_c × Recall_c / (Precision_c + Recall_c) — harmonic mean of precision and recall.
- **Macro F1** = (F1_Negative + F1_Neutral + F1_Positive) / 3.

**Edge case handling:**

`if precision + recall == 0: f1_scores.append(0.0)` — if a class has no true positives and no false positives (i.e., the model never predicts this class), F1 is 0.0. This can happen when a class is very rare in the validation set and the model learns to ignore it.

```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.asarray(logits)
    labels = np.asarray(labels)
    accuracy = float((predictions.argmax(axis=-1) == labels).mean())
    macro_f1 = compute_macro_f1(predictions, labels)
    return {"accuracy": accuracy, "macro_f1": macro_f1}
```

Both accuracy and macro_f1 are computed and logged. The `metric_for_best_model="macro_f1"` in TrainingArguments selects the best checkpoint based on macro F1. Accuracy is logged for reference but not used for checkpoint selection.

### 7.7 Model Saving and Metadata

After training completes:

```python
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
    json.dumps(metadata, indent=2), encoding="utf-8"
)
```

**Files saved to `backend/models/modernbert-sentiment/`:**

| File | Purpose |
|---|---|
| `config.json` | Model architecture configuration (num_layers, hidden_size, id2label, label2id) |
| `model.safetensors` | Best checkpoint weights in safetensors format |
| `tokenizer.json` | Tokeniser vocabulary and merge rules |
| `tokenizer_config.json` | Tokeniser settings (padding side, truncation, max length) |
| `training_args.bin` | Serialised TrainingArguments for reproducibility |
| `scc_model_metadata.json` | Dataset info, label mapping, final metrics |
| `checkpoint-2852/` | Intermediate best checkpoint (epoch N) |
| `checkpoint-4278/` | Latest best checkpoint (epoch N+1 or final) |

**`id2label` in `config.json`:**

```json
{
  "id2label": {"0": "Negative", "1": "Neutral", "2": "Positive"},
  "label2id": {"Negative": 0, "Neutral": 1, "Positive": 2}
}
```

These are essential for the HuggingFace pipeline to map model output IDs back to label names. The `_normalize_sentiment_output()` function in `main.py` then maps these label names to the canonical SCC Positive/Neutral/Negative scores.

**Backend auto-discovery:**

```python
def _get_configured_modernbert_model() -> str:
    env_value = os.getenv("MODERNBERT_SENTIMENT_MODEL", "").strip()
    if env_value:
        return env_value
    config_path = os.path.join(DEFAULT_LOCAL_MODERNBERT_PATH, "config.json")
    if os.path.exists(config_path):
        return DEFAULT_LOCAL_MODERNBERT_PATH
    return ""
```

The backend checks for `config.json` in the default local path. If found, the local checkpoint is used without any configuration change. This means running `train_modernbert_sentiment.py` with the default `--output-dir` automatically enables ModernBERT for the next server restart.

---

*End of Part 2 — Continues in Part 3*
