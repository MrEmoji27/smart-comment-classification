"""
Smart Comment Classification - Backend API v3.0
FastAPI + HuggingFace Transformers

Models:
  - Sentiment:  cardiffnlp/twitter-roberta-base-sentiment-latest (124M tweets)
  - Toxicity:   unitary/toxic-bert (Jigsaw data)
  - Type:       facebook/bart-large-mnli (zero-shot NLI)
  - Emotion:    SamLowe/roberta-base-go_emotions (28 emotions, Reddit)
  - Sarcasm:    cardiffnlp/twitter-roberta-base-irony (irony detection)
  - Word-level: VADER lexicon (supplementary)

Features:
  - Text preprocessing (emojis, slang, contractions, URLs, hashtags)
  - Gibberish/nonsense detection
  - Multi-sentence aggregation
  - Confidence thresholding
  - Spell correction (lightweight)
  - Language detection
  - Emotion-informed type classification
  - Sarcasm-aware sentiment adjustment
"""

import time
import uuid
import io
import os
import re
import html
import unicodedata
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji

# ──────────────────────────────────────
# App Init
# ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    load_model()
    yield

app = FastAPI(title="Smart Comment Classification API", version="3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────
# Rate Limiting (simple in-memory)
# ──────────────────────────────────────
rate_limit_store: dict = {}
RATE_LIMIT = 60

def check_rate_limit(client_ip: str):
    now = time.time()
    if client_ip not in rate_limit_store:
        rate_limit_store[client_ip] = []
    rate_limit_store[client_ip] = [t for t in rate_limit_store[client_ip] if now - t < 60]
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 60 requests per minute.")
    rate_limit_store[client_ip].append(now)

# ──────────────────────────────────────
# Language Detection (lightweight, no deps)
# ──────────────────────────────────────
# Common English words — if a text has very few of these, it's likely not English
COMMON_EN_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
    'this', 'but', 'his', 'by', 'from', 'they', 'we', 'her', 'she',
    'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
    'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which',
    'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
    'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good',
    'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now',
    'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
    'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well',
    'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give',
    'day', 'most', 'us', 'is', 'are', 'was', 'were', 'been', 'has',
    'had', 'did', 'does', 'am', 'very', 'much', 'too', 'more',
}

def detect_language_is_english(text: str) -> bool:
    """
    Lightweight English detection without external dependencies.
    Returns True if text appears to be English, False otherwise.
    """
    # Strip non-alpha, lowercase, split
    words = re.findall(r'[a-zA-Z]+', text.lower())
    if not words:
        # Pure numbers/symbols/emojis — treat as language-neutral (allow)
        return True
    if len(words) <= 2:
        # Too short to detect reliably — give benefit of the doubt
        return True

    en_count = sum(1 for w in words if w in COMMON_EN_WORDS)
    ratio = en_count / len(words)

    # If at least 20% of words are common English words, it's likely English.
    # This threshold is low to allow informal/slang-heavy text through.
    return ratio >= 0.15

# ──────────────────────────────────────
# Spell Correction (lightweight)
# ──────────────────────────────────────
# Common misspellings map — covers the most frequent typos that affect
# sentiment/meaning. We don't use a full spell checker (too slow, too aggressive).
COMMON_TYPOS = {
    "teh": "the", "hte": "the", "taht": "that", "recieve": "receive",
    "definately": "definitely", "seperate": "separate", "occured": "occurred",
    "untill": "until", "wierd": "weird", "alot": "a lot",
    "noone": "no one", "everytime": "every time", "thankyou": "thank you",
    "doesnt": "does not", "didnt": "did not", "isnt": "is not",
    "wasnt": "was not", "cant": "cannot", "wont": "will not",
    "dont": "do not", "ive": "i have", "youre": "you are",
    "theyre": "they are", "theres": "there is", "couldnt": "could not",
    "wouldnt": "would not", "shouldnt": "should not", "hadnt": "had not",
    "hasnt": "has not", "havent": "have not", "arent": "are not",
    "werent": "were not", "im": "i am", "youve": "you have",
    "weve": "we have", "theyve": "they have", "whats": "what is",
    "heres": "here is", "thats": "that is", "whos": "who is",
    "thier": "their", "wich": "which", "becuase": "because",
    "beacuse": "because", "thnx": "thanks", "ppl": "people",
    "msg": "message", "prob": "probably", "probs": "probably",
    "rly": "really", "rlly": "really", "sry": "sorry",
    "diff": "different", "govt": "government", "info": "information",
    "pics": "pictures", "pic": "picture", "fav": "favorite",
    "fave": "favorite", "convo": "conversation", "def": "definitely",
    "prolly": "probably", "abt": "about", "govt": "government",
    "w": "with", "b": "be", "n": "and", "r": "are", "u": "you",
    "yr": "your", "k": "okay", "luv": "love", "gud": "good",
    "gr8": "great", "h8": "hate", "l8": "late", "l8r": "later",
    "2day": "today", "2nite": "tonight", "2morrow": "tomorrow",
    "b4": "before", "coz": "because", "dat": "that", "dis": "this",
    "da": "the", "dey": "they", "dem": "them", "wat": "what",
    "wen": "when", "wer": "where", "wud": "would", "shud": "should",
    "cud": "could", "sumthing": "something", "nuthing": "nothing",
    "evrything": "everything", "evry": "every", "beutiful": "beautiful",
    "awsome": "awesome", "amazin": "amazing", "terribl": "terrible",
    "horribl": "horrible",
}

def apply_spell_correction(text: str) -> str:
    """Apply lightweight spell correction using common typo dictionary."""
    words = text.split()
    corrected = []
    for word in words:
        lower = word.lower().strip(".,!?;:'\"")
        if lower in COMMON_TYPOS:
            corrected.append(COMMON_TYPOS[lower])
        else:
            corrected.append(word)
    return ' '.join(corrected)

# ──────────────────────────────────────
# Text Preprocessing Pipeline
# ──────────────────────────────────────
CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
    "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
    "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'll": "it will", "it's": "it is", "let's": "let us",
    "might've": "might have", "must've": "must have", "mustn't": "must not",
    "needn't": "need not", "she'd": "she would", "she'll": "she will",
    "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "wasn't": "was not", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what're": "what are", "what's": "what is",
    "what've": "what have", "where's": "where is", "who'd": "who would",
    "who'll": "who will", "who's": "who is", "who've": "who have",
    "won't": "will not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have",
    "y'all": "you all", "ma'am": "madam", "o'clock": "of the clock",
    "gonna": "going to", "gotta": "got to", "wanna": "want to",
    "kinda": "kind of", "sorta": "sort of", "dunno": "do not know",
    "lemme": "let me", "gimme": "give me", "coulda": "could have",
    "shoulda": "should have", "woulda": "would have", "ima": "i am going to",
    "tryna": "trying to", "finna": "fixing to", "boutta": "about to",
}

SLANG_MAP = {
    "brb": "be right back", "btw": "by the way", "smh": "shaking my head",
    "imo": "in my opinion", "imho": "in my humble opinion",
    "tbh": "to be honest", "idk": "i do not know", "omg": "oh my god",
    "lol": "laughing out loud", "lmao": "laughing my ass off",
    "rofl": "rolling on the floor laughing", "ngl": "not going to lie",
    "iirc": "if i recall correctly", "fwiw": "for what it is worth",
    "afaik": "as far as i know", "ftw": "for the win", "wtf": "what the fuck",
    "stfu": "shut the fuck up", "nvm": "never mind", "ikr": "i know right",
    "rn": "right now", "af": "as fuck", "lowkey": "subtly",
    "highkey": "very much", "srsly": "seriously", "pls": "please",
    "plz": "please", "thx": "thanks", "ty": "thank you",
    "np": "no problem", "yw": "you are welcome", "ofc": "of course",
    "obvi": "obviously", "tho": "though", "nah": "no",
    "yep": "yes", "yup": "yes", "nope": "no",
    "cuz": "because", "bc": "because",
    "ur": "your", "2": "to", "4": "for", "m8": "mate",
    "w/": "with", "w/o": "without", "w/e": "whatever",
}

def preprocess_text(text: str) -> str:
    """Preprocess text to normalize informal English for better model understanding."""
    if not text or not text.strip():
        return text

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)

    def segment_hashtag(match):
        tag = match.group(1)
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
        words = words.replace('_', ' ')
        return words
    text = re.sub(r'#(\w+)', segment_hashtag, text)

    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'([!?.]){3,}', r'\1\1', text)

    # Spell correction before contraction expansion
    text = apply_spell_correction(text)

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

    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ──────────────────────────────────────
# Gibberish / Nonsense Detection
# ──────────────────────────────────────
def is_gibberish(text: str) -> bool:
    """Detect random keyboard mashing / nonsensical strings."""
    stripped = text.strip()
    if not stripped:
        return False

    # Pure numeric gibberish
    digits_only = re.sub(r'[\s\-_.,:;!?]', '', stripped)
    if digits_only.isdigit() and len(digits_only) >= 4:
        if len(digits_only) >= 8:
            return True
        if re.search(r'(\d{2,4})\1{1,}', digits_only):
            return True

    alpha_only = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
    if not alpha_only:
        return False

    words = alpha_only.split()
    if not words:
        return False

    if len(alpha_only.replace(' ', '')) <= 2:
        return False

    KNOWN_SHORT = {
        'ok', 'no', 'yes', 'lol', 'omg', 'wtf', 'bruh', 'nah', 'yep', 'nope',
        'hmm', 'huh', 'meh', 'ugh', 'pfft', 'shh', 'psst', 'tsk', 'wow',
        'gg', 'ez', 'oof', 'brb', 'smh', 'tbh', 'idk', 'ngl', 'fr',
        'a', 'i', 'an', 'the', 'is', 'it', 'to', 'be', 'or', 'so',
    }
    if all(w in KNOWN_SHORT for w in words):
        return False

    gibberish_words = 0
    for word in words:
        if word in KNOWN_SHORT or len(word) <= 2:
            continue

        vowels = sum(1 for c in word if c in 'aeiou')
        ratio = vowels / len(word) if len(word) > 0 else 0
        if ratio < 0.1 or ratio > 0.75:
            gibberish_words += 1
            continue
        if re.search(r'[^aeiou]{5,}', word):
            gibberish_words += 1
            continue
        if len(word) >= 6 and re.search(r'(.{2,3})\1{2,}', word):
            gibberish_words += 1
            continue

        keyboard_rows = ['qwertyuiop', 'asdfghjkl', 'zxcvbnm']
        is_keyboard_seq = False
        for row in keyboard_rows:
            for start in range(len(row) - 3):
                seq = row[start:start + 4]
                if seq in word or seq[::-1] in word:
                    is_keyboard_seq = True
                    break
            if is_keyboard_seq:
                break
        if is_keyboard_seq and len(word) >= 5:
            gibberish_words += 1
            continue

    substantive_words = [w for w in words if w not in KNOWN_SHORT and len(w) > 2]
    if not substantive_words:
        return False
    return gibberish_words / len(substantive_words) >= 0.5

# ──────────────────────────────────────
# Multi-Sentence Splitting
# ──────────────────────────────────────
def split_sentences(text: str) -> list:
    """Split text into sentences for per-sentence classification."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter empty and very short fragments
    return [s.strip() for s in sentences if len(s.strip()) >= 3]

# ──────────────────────────────────────
# Model Loading
# ──────────────────────────────────────
sentiment_classifier = None
toxicity_classifier = None
type_classifier = None
emotion_classifier = None
sarcasm_classifier = None
vader_analyzer = SentimentIntensityAnalyzer()

CANDIDATE_TYPES = [
    "asking a question or seeking information",
    "giving constructive feedback or suggestion",
    "making a complaint or expressing dissatisfaction",
    "spam or promotional or irrelevant content",
    "expressing praise or appreciation or compliment",
    "general comment or observation",
]
TYPE_LABEL_MAP = {
    "asking a question or seeking information": "Question",
    "giving constructive feedback or suggestion": "Feedback",
    "making a complaint or expressing dissatisfaction": "Complaint",
    "spam or promotional or irrelevant content": "Spam",
    "expressing praise or appreciation or compliment": "Praise",
    "general comment or observation": "Other",
}

# Emotion -> type mapping: which emotions strongly suggest which comment type
EMOTION_TYPE_BOOST = {
    # Emotions that suggest Complaint
    "anger": ("Complaint", 0.15),
    "annoyance": ("Complaint", 0.12),
    "disappointment": ("Complaint", 0.10),
    "disgust": ("Complaint", 0.12),
    # Emotions that suggest Praise
    "admiration": ("Praise", 0.15),
    "gratitude": ("Praise", 0.18),
    "joy": ("Praise", 0.10),
    "love": ("Praise", 0.12),
    "approval": ("Praise", 0.10),
    "excitement": ("Praise", 0.10),
    # Emotions that suggest Question
    "confusion": ("Question", 0.12),
    "curiosity": ("Question", 0.15),
    # Emotions that suggest Feedback
    "desire": ("Feedback", 0.08),
    "optimism": ("Feedback", 0.06),
}

def load_model():
    global sentiment_classifier, toxicity_classifier, type_classifier
    global emotion_classifier, sarcasm_classifier
    try:
        print("Loading Sentiment Model (twitter-roberta)...")
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None, device=-1
        )

        print("Loading Toxicity Model (toxic-bert)...")
        toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            top_k=None, device=-1
        )

        print("Loading Zero-Shot Type Classifier (bart-large-mnli)...")
        type_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )

        print("Loading Emotion Model (go_emotions)...")
        emotion_classifier = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=5,  # Get top 5 emotions
            device=-1
        )

        print("Loading Sarcasm/Irony Model (twitter-roberta-irony)...")
        sarcasm_classifier = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-irony",
            device=-1
        )

        print("All models loaded successfully!")
    except Exception as e:
        print(f"Critical error loading models: {e}")

# ──────────────────────────────────────
# Heuristic Boosters for Comment Type
# ──────────────────────────────────────
def apply_type_heuristics(text: str, type_scores: dict, sentiment: str,
                          tox_score: float, emotions: list = None) -> dict:
    """
    Boost zero-shot type scores with linguistic heuristics and emotion signals.
    """
    lower = text.lower().strip()
    boosted = dict(type_scores)

    # ── Emotion-based boosting ──
    if emotions:
        for emo in emotions:
            label = emo.get("label", "")
            score = emo.get("score", 0)
            if label in EMOTION_TYPE_BOOST and score > 0.3:
                target_type, boost_val = EMOTION_TYPE_BOOST[label]
                boosted[target_type] = boosted.get(target_type, 0) + boost_val * score

    # ── Question detection ──
    question_patterns = [
        r'\?',
        r'^(what|where|when|why|how|who|which|is|are|was|were|do|does|did|can|could|would|should|will|shall|has|have|had)\b',
        r'^(anybody|anyone|anything|somebody|someone|something)\b',
        r'\b(how come|what if|is it|isn\'t it|aren\'t they)\b',
    ]
    question_signal = sum(1 for p in question_patterns if re.search(p, lower))
    if question_signal >= 1:
        boosted["Question"] = boosted.get("Question", 0) + 0.25 * question_signal

    # ── Uncertainty / ambivalence ──
    uncertainty_patterns = r'\b(idk|i do not know|not sure|unsure|uncertain|maybe|might|on the fence|hard to say|can\'t decide|cannot decide|i guess|idk if|not sure if|don\'t know if|do not know if)\b'
    if re.search(uncertainty_patterns, lower):
        boosted["Other"] = boosted.get("Other", 0) + 0.15
        boosted["Complaint"] = boosted.get("Complaint", 0) * 0.5

    # ── Complaint signals ──
    complaint_words = r'\b(terrible|horrible|awful|worst|hate|sucks|broken|unusable|unacceptable|ridiculous|disgusting|pathetic|waste|scam|fraud|rip.?off|garbage|trash|useless|disappointed|frustrating|infuriating)\b'
    if re.search(complaint_words, lower):
        boosted["Complaint"] = boosted.get("Complaint", 0) + 0.2
    if sentiment == "Negative" and tox_score > 0.3:
        boosted["Complaint"] = boosted.get("Complaint", 0) + 0.15

    superlative_complaint = r'\b(stupidest|dumbest|worst|most annoying|most useless|most pathetic|most ridiculous)\b.*\b(ever|in my life|of all time|i have ever|i\'ve ever)\b'
    if re.search(superlative_complaint, lower):
        boosted["Complaint"] = boosted.get("Complaint", 0) + 0.4

    # ── Praise signals ──
    praise_words = r'\b(amazing|awesome|excellent|fantastic|wonderful|brilliant|outstanding|perfect|incredible|love|loved|loving|best|great|superb|magnificent|phenomenal|thank|thanks|grateful|impressed)\b'
    if re.search(praise_words, lower):
        boosted["Praise"] = boosted.get("Praise", 0) + 0.2
    if sentiment == "Positive":
        boosted["Praise"] = boosted.get("Praise", 0) + 0.1

    slang_praise_words = r'\b(fire|goat|lit|sick|bussin|bussing|slaps|slap|dope|heat|chef\'s kiss|elite|insane|valid|hard|goes hard|hits different|peak|god.?tier|next level|top.?tier|no cap|on point|clean|mint|ace|bangin|banging|killer|banger)\b'
    if re.search(slang_praise_words, lower):
        boosted["Praise"] = boosted.get("Praise", 0) + 0.35
        boosted["Complaint"] = boosted.get("Complaint", 0) * 0.4

    # ── Feedback signals ──
    feedback_words = r'\b(should|could|would be better|suggest|suggestion|recommend|consider|improve|improvement|perhaps|it would be nice|feature request|adding|please add|wish|hope)\b'
    if re.search(feedback_words, lower):
        boosted["Feedback"] = boosted.get("Feedback", 0) + 0.2

    # ── Spam signals (multiplicative) ──
    spam_multiplier = 1.0

    spam_keywords = r'\b(buy now|click here|free|discount|offer|promo|subscribe|check out my|follow me|visit|earn money|make money|limited time|act now|order now|deal|sale|giveaway|giveaways|win a|winner|signup|sign up)\b'
    spam_keyword_matches = len(re.findall(spam_keywords, lower))
    if spam_keyword_matches >= 1:
        spam_multiplier *= (1.0 + 0.4 * spam_keyword_matches)

    if re.search(r'(www\.|\.com|\.net|\.org|\.io|https?://)', lower):
        spam_multiplier *= 1.8

    exclamation_count = len(re.findall(r'!', text))
    if exclamation_count >= 2:
        spam_multiplier *= (1.0 + 0.2 * min(exclamation_count, 5))

    caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
    if len(caps_words) >= 1:
        spam_multiplier *= (1.0 + 0.3 * min(len(caps_words), 5))

    imperative_cmds = r'(?:^|[.!?]\s*)(buy|shop|click|visit|follow|subscribe|join|sign up|check out|order|call|get your|grab|hurry|act now|don\'t miss)\b'
    if re.search(imperative_cmds, lower):
        spam_multiplier *= 1.5

    if spam_multiplier > 1.0:
        base_spam = max(boosted.get("Spam", 0), 0.08)
        boosted["Spam"] = base_spam * spam_multiplier
        if spam_multiplier >= 2.5:
            boosted["Praise"] = boosted.get("Praise", 0) * 0.3

    # ── Normalize ──
    total = sum(boosted.values())
    if total > 0:
        boosted = {k: round(v / total, 4) for k, v in boosted.items()}

    return boosted

# ──────────────────────────────────────
# Confidence Thresholding
# ──────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55

def apply_confidence_flag(sentiment: str, sent_scores: dict) -> tuple:
    """
    If the top sentiment confidence is below threshold, flag as uncertain.
    Returns (sentiment_label, is_uncertain).
    """
    max_conf = max(sent_scores.values())
    is_uncertain = max_conf < CONFIDENCE_THRESHOLD
    return sentiment, is_uncertain

# ──────────────────────────────────────
# Multi-Sentence Aggregation
# ──────────────────────────────────────
def classify_multi_sentence(text: str, cleaned: str) -> dict:
    """
    For multi-sentence text, classify each sentence individually and aggregate.
    Returns per-sentence breakdown and overall aggregated scores.
    """
    sentences = split_sentences(cleaned)

    # If single sentence or very short, skip aggregation
    if len(sentences) <= 1:
        return None

    per_sentence = []
    agg_scores = {"Positive": 0, "Neutral": 0, "Negative": 0}

    for sent_text in sentences:
        try:
            result = sentiment_classifier(sent_text[:512])
            scores = {}
            for item in result[0] if isinstance(result[0], list) else result:
                label = item["label"].lower()
                if "positive" in label:
                    mapped = "Positive"
                elif "negative" in label:
                    mapped = "Negative"
                else:
                    mapped = "Neutral"
                scores[mapped] = scores.get(mapped, 0) + item["score"]

            total = sum(scores.values())
            if total > 0:
                scores = {k: round(v / total, 4) for k, v in scores.items()}
            for k in ["Positive", "Neutral", "Negative"]:
                if k not in scores:
                    scores[k] = 0.0

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

    # Average the scores
    n = len(per_sentence)
    if n > 0:
        agg_scores = {k: round(v / n, 4) for k, v in agg_scores.items()}

    return {
        "sentences": per_sentence,
        "aggregated_scores": agg_scores,
        "aggregated_sentiment": max(agg_scores, key=agg_scores.get),
        "is_mixed": len(set(s["sentiment"] for s in per_sentence)) > 1
    }

# ──────────────────────────────────────
# Helper: classify single text
# ──────────────────────────────────────
def classify_text_internal(text: str) -> dict:
    if sentiment_classifier is None or toxicity_classifier is None or type_classifier is None:
        raise HTTPException(status_code=503, detail="Models not fully loaded. Please try again later.")

    start = time.time()

    # Check for gibberish
    gibberish = is_gibberish(text)

    # Check language
    is_english = detect_language_is_english(text)

    # Preprocess
    cleaned = preprocess_text(text)
    truncated = cleaned[:2000]

    # 1. Sentiment
    sent_results = sentiment_classifier(truncated)
    sent_scores = {}
    for item in sent_results[0] if isinstance(sent_results[0], list) else sent_results:
        label = item["label"].lower()
        if "positive" in label:
            mapped = "Positive"
        elif "negative" in label:
            mapped = "Negative"
        else:
            mapped = "Neutral"
        sent_scores[mapped] = sent_scores.get(mapped, 0) + item["score"]

    total = sum(sent_scores.values())
    if total > 0:
        sent_scores = {k: round(v / total, 4) for k, v in sent_scores.items()}
    for k in ["Positive", "Neutral", "Negative"]:
        if k not in sent_scores:
            sent_scores[k] = 0.0
    predicted_sentiment = max(sent_scores, key=sent_scores.get)

    # 2. Sarcasm/Irony Detection
    is_sarcastic = False
    sarcasm_score = 0.0
    if sarcasm_classifier:
        try:
            sarc_result = sarcasm_classifier(truncated[:512])
            for item in sarc_result if isinstance(sarc_result, list) else [sarc_result]:
                if isinstance(item, list):
                    item = item[0]
                if item["label"].lower() in ("irony", "label_1", "1"):
                    sarcasm_score = item["score"]
        except Exception:
            pass

    # Sarcasm requires both: high model score AND negative contextual cues
    # (prevents false positives on genuinely enthusiastic text)
    lower_text = text.lower()
    has_negative_cues = bool(re.search(
        r'\b(crash|broke|broken|fail|problem|issue|wrong|bad|worst|terrible|horrible|sucks|hate)\b'
        r'|(\d+\s*times)',
        lower_text
    ))
    # Gratitude phrases suppress sarcasm detection
    has_gratitude = bool(re.search(
        r'\b(thanks?|thx|ty|tysm|thank you|grateful|appreciate|thats? great|good job|nice work|well done|awesome work)\b',
        lower_text
    ))
    if has_gratitude:
        has_negative_cues = False

    # Only flag sarcasm if model is very confident AND text has negative context
    if sarcasm_score > 0.80 and has_negative_cues:
        is_sarcastic = True

    # If sarcastic and model says positive, it's likely actually negative
    if is_sarcastic and predicted_sentiment == "Positive":
        predicted_sentiment = "Negative"
        old_pos = sent_scores.get("Positive", 0)
        old_neg = sent_scores.get("Negative", 0)
        sent_scores["Positive"] = round(old_neg * 0.5, 4)
        sent_scores["Negative"] = round(old_pos * 0.8, 4)
        sent_scores["Neutral"] = round(1.0 - sent_scores["Positive"] - sent_scores["Negative"], 4)

    # 3. Toxicity
    tox_results = toxicity_classifier(truncated)
    tox_score = 0.0
    for item in tox_results[0] if isinstance(tox_results[0], list) else tox_results:
        if item["label"].lower() == "toxic":
            tox_score = item["score"]
    is_toxic = tox_score > 0.5

    # 4. Emotion Detection
    emotions = []
    if emotion_classifier:
        try:
            emo_results = emotion_classifier(truncated[:512])
            if isinstance(emo_results, list) and len(emo_results) > 0:
                if isinstance(emo_results[0], list):
                    emotions = emo_results[0]
                else:
                    emotions = emo_results
        except Exception:
            pass

    # 5. Comment Type (zero-shot + heuristics + emotion boosting)
    type_results = type_classifier(truncated, candidate_labels=CANDIDATE_TYPES)
    raw_type_scores = {}
    for label, score in zip(type_results["labels"], type_results["scores"]):
        clean_label = TYPE_LABEL_MAP.get(label, label)
        raw_type_scores[clean_label] = round(score, 4)

    type_scores = apply_type_heuristics(
        text, raw_type_scores, predicted_sentiment, tox_score, emotions
    )

    # If sarcasm detected and sentiment was flipped to Negative, boost Complaint over Praise
    if is_sarcastic and predicted_sentiment == "Negative":
        praise_val = type_scores.get("Praise", 0)
        type_scores["Complaint"] = type_scores.get("Complaint", 0) + praise_val * 0.6
        type_scores["Praise"] = praise_val * 0.3
        # Re-normalize
        total_t = sum(type_scores.values())
        if total_t > 0:
            type_scores = {k: round(v / total_t, 4) for k, v in type_scores.items()}

    predicted_type = max(type_scores, key=type_scores.get)

    # 6. Confidence thresholding
    predicted_sentiment, is_uncertain = apply_confidence_flag(predicted_sentiment, sent_scores)

    # 7. Multi-sentence analysis (only for 2+ sentence text)
    multi_sentence = classify_multi_sentence(text, cleaned)
    if multi_sentence and multi_sentence["is_mixed"]:
        # Use aggregated scores for mixed-sentiment text
        sent_scores = multi_sentence["aggregated_scores"]
        predicted_sentiment = multi_sentence["aggregated_sentiment"]
        predicted_sentiment, is_uncertain = apply_confidence_flag(predicted_sentiment, sent_scores)

    # Override for gibberish
    if gibberish:
        predicted_sentiment = "Neutral"
        sent_scores = {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0}
        predicted_type = "Spam"
        type_scores = {k: (1.0 if k == "Spam" else 0.0) for k in type_scores}
        is_uncertain = False
        emotions = []
        is_sarcastic = False

    latency_ms = round((time.time() - start) * 1000)

    # 8. Word-Level Lexicon Sentiments (VADER on original text)
    tokens = re.findall(r'\S+|\s+', text)
    word_analysis = []
    word_counts = {"total": 0, "positive": 0, "neutral": 0, "negative": 0}

    words_only = [t for t in tokens if t.strip()]
    for i, token in enumerate(tokens):
        if token.strip():
            word_idx = words_only.index(token) if token in words_only else -1
            if word_idx >= 0:
                context_start = max(0, word_idx - 2)
                context_words = words_only[context_start:word_idx + 1]
                context_phrase = ' '.join(context_words)
                if len(context_words) > 1:
                    phrase_score = vader_analyzer.polarity_scores(context_phrase)['compound']
                    single_score = vader_analyzer.polarity_scores(token)['compound']
                    if (phrase_score > 0 and single_score < 0) or (phrase_score < 0 and single_score > 0):
                        score = phrase_score
                    else:
                        score = single_score
                else:
                    score = vader_analyzer.polarity_scores(token)['compound']
            else:
                score = vader_analyzer.polarity_scores(token)['compound']

            if score >= 0.05:
                sent = "Positive"
            elif score <= -0.05:
                sent = "Negative"
            else:
                sent = "Neutral"
            word_analysis.append({"text": token, "sentiment": sent})
            word_counts["total"] += 1
            word_counts[sent.lower()] += 1
        else:
            word_analysis.append({"text": token, "sentiment": "Whitespace"})

    # Build response
    response = {
        "sentiment": predicted_sentiment,
        "sentiment_confidence": {
            "positive": sent_scores.get("Positive", 0),
            "neutral": sent_scores.get("Neutral", 0),
            "negative": sent_scores.get("Negative", 0),
        },
        "is_uncertain": is_uncertain,
        "toxicity": round(tox_score, 4),
        "is_toxic": is_toxic,
        "comment_type": predicted_type,
        "type_scores": type_scores,
        "is_sarcastic": is_sarcastic,
        "sarcasm_score": round(sarcasm_score, 4),
        "emotions": [{"label": e["label"], "score": round(e["score"], 4)} for e in emotions[:5]] if emotions else [],
        "is_english": is_english,
        "word_analysis": word_analysis,
        "word_counts": word_counts,
        "latency_ms": latency_ms,
    }

    # Include per-sentence breakdown if multi-sentence
    if multi_sentence:
        response["multi_sentence"] = {
            "sentences": multi_sentence["sentences"],
            "is_mixed": multi_sentence["is_mixed"],
        }

    return response

# ──────────────────────────────────────
# Job Store (in-memory)
# ──────────────────────────────────────
jobs: dict = {}

def process_batch_job(job_id: str, texts: list):
    """Background task: classify a list of texts and update job store."""
    job = jobs[job_id]
    job["status"] = "processing"
    results = []

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for text in batch:
            try:
                result = classify_text_internal(str(text))
                results.append({
                    "comment": str(text),
                    "sentiment": result["sentiment"],
                    "conf_pos": result["sentiment_confidence"]["positive"],
                    "conf_neu": result["sentiment_confidence"]["neutral"],
                    "conf_neg": result["sentiment_confidence"]["negative"],
                    "toxicity": result["toxicity"],
                    "is_toxic": result["is_toxic"],
                    "comment_type": result["comment_type"],
                    "is_sarcastic": result.get("is_sarcastic", False),
                    "is_uncertain": result.get("is_uncertain", False),
                    "emotions": result.get("emotions", []),
                    "word_analysis": result.get("word_analysis", []),
                    "word_counts": result.get("word_counts", {})
                })
            except Exception as e:
                print(f"Error on batch item: {e}")
                results.append({
                    "comment": str(text),
                    "sentiment": "Error",
                    "conf_pos": 0, "conf_neu": 0, "conf_neg": 0,
                    "toxicity": 0, "is_toxic": False, "comment_type": "Unknown",
                    "is_sarcastic": False, "is_uncertain": False,
                    "emotions": [],
                    "word_analysis": [], "word_counts": {}
                })
            job["processed"] = len(results)

    job["status"] = "done"
    job["results"] = results

# ──────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok" if (sentiment_classifier and toxicity_classifier and type_classifier) else "model_not_loaded",
        "sentiment_loaded": sentiment_classifier is not None,
        "toxicity_loaded": toxicity_classifier is not None,
        "type_loaded": type_classifier is not None,
        "emotion_loaded": emotion_classifier is not None,
        "sarcasm_loaded": sarcasm_classifier is not None,
        "version": "3.0"
    }

@app.post("/classify/text")
async def classify_text_endpoint(request: Request):
    check_rate_limit(request.client.host)

    body = await request.json()
    text = body.get("text", "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Please enter a comment. Text field cannot be empty.")
    if len(text) > 8192:
        raise HTTPException(status_code=400, detail="Text exceeds maximum length of 8192 characters.")

    result = classify_text_internal(text)
    return JSONResponse(content=result)

@app.post("/classify/file")
async def classify_file_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    column: Optional[str] = Form(None)
):
    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in [".csv", ".txt", ".xlsx"]):
        raise HTTPException(status_code=400, detail="Only .csv, .txt, .xlsx files are accepted.")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds maximum size of 10MB.")

    texts = []
    columns_list = []
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            columns_list = df.columns.tolist()
            if column and column in df.columns:
                texts = df[column].dropna().tolist()
            elif len(df.columns) == 1:
                texts = df.iloc[:, 0].dropna().tolist()
            else:
                return JSONResponse(content={
                    "status": "needs_column",
                    "columns": columns_list,
                    "message": "Please select the text column to classify."
                })
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
            columns_list = df.columns.tolist()
            if column and column in df.columns:
                texts = df[column].dropna().tolist()
            elif len(df.columns) == 1:
                texts = df.iloc[:, 0].dropna().tolist()
            else:
                return JSONResponse(content={
                    "status": "needs_column",
                    "columns": columns_list,
                    "message": "Please select the text column to classify."
                })
        elif filename.endswith(".txt"):
            text_content = content.decode("utf-8", errors="replace")
            texts = [line.strip() for line in text_content.split("\n") if line.strip()]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")

    if not texts:
        raise HTTPException(status_code=400, detail="File appears to be empty. No text data found.")
    if len(texts) > 5000:
        texts = texts[:5000]

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "total": len(texts),
        "processed": 0,
        "results": []
    }

    background_tasks.add_task(process_batch_job, job_id, texts)

    return JSONResponse(content={
        "job_id": job_id,
        "status": "processing",
        "total_rows": len(texts)
    })

@app.get("/classify/status/{job_id}")
async def classify_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]
    response = {
        "status": job["status"],
        "processed": job["processed"],
        "total": job["total"],
    }

    if job["status"] == "done":
        response["results"] = job["results"]

    return JSONResponse(content=response)

# ──────────────────────────────────────
# Run
# ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
