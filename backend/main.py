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
import logging
import os
import re
import html
import unicodedata
from collections import Counter
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji

LOGGER = logging.getLogger("smart_comment_classification")
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

APP_VERSION = "3.2"
DEFAULT_SENTIMENT_FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_LOCAL_MODERNBERT_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "modernbert-sentiment",
)

# ──────────────────────────────────────
# App Init
# ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    load_model()
    yield

app = FastAPI(title="Smart Comment Classification API", version=APP_VERSION, lifespan=lifespan)

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
    "prolly": "probably", "abt": "about",
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
# Context-Aware Input Building
# ──────────────────────────────────────
_CONTEXT_SEP = " [SEP] "

def build_contextual_input(text: str, context: str) -> str:
    """
    Prepend surrounding context to text so the model sees both during inference.
    Format: "[Context: {context}] [SEP] {text}"
    This is a standard prompting technique for transformers — the model attends
    across the full context+text sequence.
    """
    if not context:
        return text
    return f"[Context: {context}]{_CONTEXT_SEP}{text}"

# Patterns for context signal detection
_CONTRAST_RE = re.compile(
    r'\b(but|however|although|though|despite|yet|still|nevertheless|nonetheless|'
    r'that said|even so|on the other hand|while|whereas|even though|in spite of|'
    r'having said that|at the same time|then again|all the same)\b',
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(not|no|never|neither|nor|cannot|can't|won't|wouldn't|shouldn't|couldn't|"
    r"isn't|aren't|wasn't|weren't|doesn't|don't|didn't|hardly|barely|scarcely|"
    r"nothing|nobody|nowhere|seldom|rarely)\b",
    re.IGNORECASE,
)
_CONDITIONAL_RE = re.compile(
    r'\b(if|unless|provided that|as long as|only if|in case|assuming|suppose|'
    r'supposing|given that|on condition that)\b',
    re.IGNORECASE,
)
_COMPARATIVE_RE = re.compile(
    r'\b(better than|worse than|more than|less than|superior to|inferior to|'
    r'compared to|compared with|relative to|not as good as|far better|much worse|'
    r'way better|way worse)\b',
    re.IGNORECASE,
)

def detect_context_signals(text: str) -> dict:
    """
    Detect linguistic patterns that affect how sentiment should be interpreted.
    Surfaces these in the API response so consumers understand *why* a label
    was assigned (e.g. negation flipped the polarity, contrast split the meaning).
    """
    return {
        "has_negation": bool(_NEGATION_RE.search(text)),
        "has_contrast": bool(_CONTRAST_RE.search(text)),
        "is_conditional": bool(_CONDITIONAL_RE.search(text)),
        "is_comparative": bool(_COMPARATIVE_RE.search(text)),
    }

# ──────────────────────────────────────
# Sensitive Topic Detection
# ──────────────────────────────────────
SENSITIVE_TOPIC_PATTERNS: dict = {
    "weapons_firearms": re.compile(
        r'\b(gun|guns|firearm|firearms|rifle|pistol|shotgun|revolver|weapon|weapons|'
        r'knife|knives|blade|ammo|ammunition|bullet|bullets|armed|'
        r'second.?amendment|conceal.?carry|open.?carry|ar-?15|ak-?47)\b',
        re.IGNORECASE,
    ),
    "politics": re.compile(
        r'\b(democrat|republican|liberal|conservative|socialist|communist|fascist|'
        r'election|vote|voting|ballot|congress|senate|parliament|president|'
        r'prime minister|politician|government|political|policy|legislation|'
        r'immigration|abortion|left.?wing|right.?wing|constitution|regime)\b',
        re.IGNORECASE,
    ),
    "religion": re.compile(
        r'\b(god|allah|jesus|christ|christian|muslim|jewish|hindu|buddhist|sikh|'
        r'religion|religious|church|mosque|temple|synagogue|prayer|pray|faith|'
        r'bible|quran|torah|atheist|agnostic|blasphemy|sacred|holy|divine)\b',
        re.IGNORECASE,
    ),
    "drugs_alcohol": re.compile(
        r'\b(drug|drugs|marijuana|weed|cannabis|cocaine|heroin|meth|methamphetamine|'
        r'alcohol|beer|wine|liquor|drunk|substance|overdose|addiction|addicted|'
        r'narcotic|opioid|fentanyl|crack|ecstasy|lsd|psychedelic)\b',
        re.IGNORECASE,
    ),
    "violence": re.compile(
        r'\b(kill|killing|killed|murder|murdered|fight|fighting|war|warfare|'
        r'violent|violence|assault|abuse|harm|death|dead|dying|genocide|'
        r'torture|bomb|terrorist|terrorism|massacre|shooting|stabbing)\b',
        re.IGNORECASE,
    ),
    "mental_health": re.compile(
        r'\b(depression|depressed|anxiety|anxious|suicide|suicidal|self.harm|'
        r'mental.health|bipolar|schizophrenia|trauma|ptsd|panic.attack|'
        r'eating.disorder|anorexia|bulimia)\b',
        re.IGNORECASE,
    ),
    "race_discrimination": re.compile(
        r'\b(racist|racism|racial|white.supremac|hate.crime|discrimination|'
        r'segregation|xenophobia|islamophobia|antisemit|ethnic.cleansing|'
        r'bigot|bigotry|slur)\b',
        re.IGNORECASE,
    ),
}

def detect_sensitive_topics(text: str) -> dict:
    """
    Detect if the comment touches on sensitive or controversial topics.
    This does NOT change the sentiment label — it adds context so consumers
    know to interpret a 'Positive' label as 'positive opinion about weapons'
    rather than a neutral cheerful statement.
    """
    detected = [topic for topic, pattern in SENSITIVE_TOPIC_PATTERNS.items() if pattern.search(text)]
    return {
        "topics": detected,
        "is_sensitive": len(detected) > 0,
    }

# ──────────────────────────────────────
# Subjectivity, Intensity & Calibration
# ──────────────────────────────────────
_OPINION_MARKERS = re.compile(
    r'\b(i think|i believe|i feel|i reckon|i guess|in my opinion|imo|imho|'
    r'personally|from my perspective|it seems to me|as far as i can tell|'
    r'i would say|to me it|for me it|my view|my opinion|my take|'
    r'i consider|i find it|i find this)\b',
    re.IGNORECASE,
)
_HEDGING_MARKERS = re.compile(
    r'\b(maybe|perhaps|possibly|probably|might|could be|seems like|kind of|'
    r'sort of|somewhat|rather|fairly|relatively|not sure if|apparently|'
    r'supposedly|allegedly|i guess|i suppose)\b',
    re.IGNORECASE,
)
_INTENSIFIER_MARKERS = re.compile(
    r'\b(very|extremely|absolutely|completely|totally|utterly|incredibly|'
    r'amazingly|terribly|horribly|really|so much|such a|way too|way more|'
    r'way less|so damn|freaking|genuinely|truly|deeply|insanely|ridiculously)\b',
    re.IGNORECASE,
)
_FACTUAL_MARKERS = re.compile(
    r'\b(according to|research shows|studies show|data shows|it is a fact|'
    r'scientifically|statistically|proven|evidence suggests|report says|'
    r'survey found|analysis shows|statistics show|experts say|facts show)\b',
    re.IGNORECASE,
)
_RHETORICAL_Q_RE = re.compile(
    r'\b(seriously\?|really\?|are you kidding|is this a joke|how is that|'
    r'why would|why on earth|what were they thinking|who does that|'
    r'can you believe|how could|is it really|do they really)\b',
    re.IGNORECASE,
)

def compute_subjectivity(text: str) -> dict:
    """
    Score how subjective vs objective the text is using linguistic markers.
    Returns a label (subjective / objective / mixed), a 0-1 score, and the
    detected markers so the frontend can explain the classification.
    """
    has_opinion = bool(_OPINION_MARKERS.search(text))
    has_hedging = bool(_HEDGING_MARKERS.search(text))
    has_intensifier = bool(_INTENSIFIER_MARKERS.search(text))
    has_factual = bool(_FACTUAL_MARKERS.search(text))
    has_rhetorical = bool(_RHETORICAL_Q_RE.search(text))

    score = 0.5  # baseline: ambiguous
    if has_opinion:
        score += 0.30
    if has_intensifier:
        score += 0.15
    if has_hedging:
        score += 0.10
    if has_rhetorical:
        score += 0.10
    if has_factual:
        score -= 0.35
    score = round(max(0.0, min(1.0, score)), 3)

    if score >= 0.65:
        label = "subjective"
    elif score <= 0.35:
        label = "objective"
    else:
        label = "mixed"

    return {
        "label": label,
        "score": score,
        "has_opinion_marker": has_opinion,
        "has_hedging": has_hedging,
        "has_intensifier": has_intensifier,
        "has_rhetorical_question": has_rhetorical,
    }

def compute_sentiment_intensity(sentiment: str, scores: dict, text: str) -> str:
    """
    Classify how strongly the sentiment is expressed.
    Combines model confidence with presence of linguistic intensifiers.
    """
    conf = scores.get(sentiment, 0.0)
    has_intensifier = bool(_INTENSIFIER_MARKERS.search(text))
    if conf >= 0.85 or (conf >= 0.75 and has_intensifier):
        return "strong"
    elif conf >= 0.65:
        return "moderate"
    elif conf >= 0.55:
        return "mild"
    else:
        return "uncertain"

def apply_rhetorical_question_adjustment(
    text: str,
    predicted_sentiment: str,
    sent_scores: dict,
    heuristics: list,
) -> tuple:
    """
    Rhetorical questions ("Are guns really safe??", "How could they do this?")
    carry implicit negative or sarcastic sentiment that the model often misses
    because the surface words may not contain explicit negative markers.
    """
    if not _RHETORICAL_Q_RE.search(text):
        return predicted_sentiment, sent_scores

    # Rhetorical questions typically express frustration/disbelief → negative lean
    sent_scores = dict(sent_scores)
    if predicted_sentiment == "Positive" or predicted_sentiment == "Neutral":
        boost = 0.15
        sent_scores["Negative"] = min(1.0, sent_scores.get("Negative", 0.0) + boost)
        total = sum(sent_scores.values())
        if total > 0:
            sent_scores = {k: round(v / total, 4) for k, v in sent_scores.items()}
        new_pred = max(sent_scores, key=sent_scores.get)
        if new_pred != predicted_sentiment:
            heuristics.append(f"rhetorical_q_adjustment:{predicted_sentiment}->{new_pred}")
            predicted_sentiment = new_pred
        else:
            heuristics.append("rhetorical_q_negative_lean")
    return predicted_sentiment, sent_scores

def calibrate_with_vader(
    text: str,
    predicted_sentiment: str,
    sent_scores: dict,
    heuristics: list,
) -> tuple:
    """
    Cross-check transformer output against VADER's rule-based lexicon scores.

    VADER is especially reliable for:
    - Clear lexical polarity ("guns are GREAT" vs "guns are terrible")
    - Short texts where transformer models have less signal
    - Texts with intensifiers (very, extremely, absolutely)

    Only intervenes when transformer confidence is below 0.65 AND VADER
    strongly disagrees (compound > 0.4 or < -0.4). This avoids overriding
    confident transformer predictions with rule-based noise.
    """
    vader_scores = vader_analyzer.polarity_scores(text)
    compound = vader_scores["compound"]

    if compound >= 0.05:
        vader_label = "Positive"
    elif compound <= -0.05:
        vader_label = "Negative"
    else:
        vader_label = "Neutral"

    transformer_conf = sent_scores.get(predicted_sentiment, 0.0)

    # Short texts (< 6 words): VADER gets more influence since transformer has less context
    word_count = len(text.split())
    vader_weight = 0.18 if word_count < 6 else 0.12

    if vader_label != predicted_sentiment and transformer_conf < 0.65 and abs(compound) > 0.4:
        sent_scores = dict(sent_scores)
        blend_amount = vader_weight * abs(compound)
        sent_scores[vader_label] = min(1.0, sent_scores.get(vader_label, 0.0) + blend_amount)
        sent_scores[predicted_sentiment] = max(0.0, sent_scores.get(predicted_sentiment, 0.0) - blend_amount * 0.5)
        total = sum(sent_scores.values())
        if total > 0:
            sent_scores = {k: round(v / total, 4) for k, v in sent_scores.items()}
        new_pred = max(sent_scores, key=sent_scores.get)
        if new_pred != predicted_sentiment:
            heuristics.append(f"vader_calibration:{predicted_sentiment}->{new_pred}")
            predicted_sentiment = new_pred
        else:
            heuristics.append("vader_confidence_blend")

    return predicted_sentiment, sent_scores

# ──────────────────────────────────────
# Model Loading
# ──────────────────────────────────────
sentiment_classifier = None
toxicity_classifier = None
type_classifier = None
emotion_classifier = None
sarcasm_classifier = None
vader_analyzer = SentimentIntensityAnalyzer()
model_tokenizers: dict = {}
model_registry: dict = {}
model_status: dict = {}

PIPELINE_DEVICE = 0 if torch.cuda.is_available() else -1
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

def _get_primary_model_identifier(name: str) -> str:
    if name == "sentiment":
        return _get_configured_modernbert_model() or DEFAULT_SENTIMENT_FALLBACK_MODEL
    spec = MODEL_SPECS[name]
    if "model" in spec:
        return spec["model"]
    for candidate in spec.get("candidates", []):
        if candidate.get("enabled") and candidate.get("model"):
            return candidate["model"]
    return ""

def _get_configured_modernbert_model() -> str:
    env_value = os.getenv("MODERNBERT_SENTIMENT_MODEL", "").strip()
    if env_value:
        return env_value

    config_path = os.path.join(DEFAULT_LOCAL_MODERNBERT_PATH, "config.json")
    if os.path.exists(config_path):
        return DEFAULT_LOCAL_MODERNBERT_PATH
    return ""

def _resolve_model_candidates(name: str) -> list[dict]:
    spec = MODEL_SPECS[name]
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
    if "candidates" in spec:
        return [
            candidate for candidate in spec["candidates"]
            if candidate.get("enabled") and candidate.get("model")
        ]
    return [{
        "name": name,
        "model": spec["model"],
        "display_name": spec.get("display_name", spec["model"]),
        "enabled": True,
    }]

def _set_model_availability(
    name: str,
    classifier,
    error: Optional[str] = None,
    actual_model: Optional[str] = None,
    display_name: Optional[str] = None,
    attempted_models: Optional[list[str]] = None,
):
    spec = MODEL_SPECS[name]
    model_status[name] = {
        "loaded": classifier is not None,
        "required": spec["required"],
        "model": actual_model or _get_primary_model_identifier(name),
        "display_name": display_name or spec.get("display_name") or (actual_model or _get_primary_model_identifier(name)),
        "max_tokens": spec["max_tokens"],
        "error": error,
        "attempted_models": attempted_models or [],
    }

def _load_pipeline(name: str):
    spec = MODEL_SPECS[name]
    attempted = []
    last_error = None

    for candidate in _resolve_model_candidates(name):
        if not candidate.get("enabled", True) or not candidate.get("model"):
            continue  # skip disabled / unconfigured candidates (e.g. ModernBERT when path not set)
        attempted.append(candidate["model"])
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate["model"], use_fast=True)
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
            _set_model_availability(
                name,
                classifier,
                actual_model=candidate["model"],
                display_name=candidate.get("display_name", candidate["model"]),
                attempted_models=attempted,
            )
            return classifier
        except Exception as exc:
            last_error = exc
            LOGGER.warning("Candidate model failed for %s: %s", name, candidate["model"])

    raise RuntimeError(f"Unable to load {name} from candidates {attempted}: {last_error}")

def truncate_for_model(text: str, model_name: str) -> tuple[str, dict]:
    tokenizer = model_tokenizers.get(model_name)
    spec = MODEL_SPECS[model_name]
    if not text or tokenizer is None:
        return text, {"truncated": False, "max_tokens": spec["max_tokens"], "input_tokens": 0}

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

def _normalize_sentiment_output(result) -> dict:
    scores = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    items = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result
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
    else:
        # All-zero scores (degenerate model output) — default to Neutral
        scores = {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0}
    return scores

def _extract_label_score(result, positive_labels: set[str]) -> float:
    items = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result
    for item in items:
        if item["label"].lower() in positive_labels:
            return item["score"]
    return 0.0

def _run_stage_batch(model_name: str, texts: list[str], **kwargs) -> tuple[list, float]:
    classifier = model_registry.get(model_name)
    if classifier is None:
        return [None] * len(texts), 0.0
    if not texts:
        return [], 0.0

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

def load_model():
    global sentiment_classifier, toxicity_classifier, type_classifier
    global emotion_classifier, sarcasm_classifier
    LOGGER.info("Loading NLP model stack on device=%s", PIPELINE_DEVICE)
    for model_name in MODEL_SPECS:
        try:
            LOGGER.info("Loading %s model candidates: %s", model_name, ", ".join(
                candidate["model"] for candidate in _resolve_model_candidates(model_name)
            ) or _get_primary_model_identifier(model_name))
            _load_pipeline(model_name)
        except Exception as exc:
            LOGGER.exception("Failed to load %s model", model_name)
            model_registry[model_name] = None
            model_tokenizers.pop(model_name, None)
            _set_model_availability(
                model_name,
                None,
                str(exc),
                attempted_models=[candidate["model"] for candidate in _resolve_model_candidates(model_name)],
            )

    sentiment_classifier = model_registry.get("sentiment")
    toxicity_classifier = model_registry.get("toxicity")
    type_classifier = model_registry.get("type")
    emotion_classifier = model_registry.get("emotion")
    sarcasm_classifier = model_registry.get("sarcasm")

    missing_required = [
        name for name, status in model_status.items()
        if status["required"] and not status["loaded"]
    ]
    if missing_required:
        LOGGER.error("Required models unavailable: %s", ", ".join(missing_required))
    else:
        LOGGER.info("Required models loaded successfully")

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
    heuristic_flags = []

    # ── Emotion-based boosting ──
    if emotions:
        for emo in emotions:
            label = emo.get("label", "")
            score = emo.get("score", 0)
            if label in EMOTION_TYPE_BOOST and score > 0.3:
                target_type, boost_val = EMOTION_TYPE_BOOST[label]
                boosted[target_type] = boosted.get(target_type, 0) + boost_val * score
                heuristic_flags.append(f"emotion:{label}->{target_type}")

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
        heuristic_flags.append("question_pattern")

    # ── Uncertainty / ambivalence ──
    uncertainty_patterns = r'\b(idk|i do not know|not sure|unsure|uncertain|maybe|might|on the fence|hard to say|can\'t decide|cannot decide|i guess|idk if|not sure if|don\'t know if|do not know if)\b'
    if re.search(uncertainty_patterns, lower):
        boosted["Other"] = boosted.get("Other", 0) + 0.15
        boosted["Complaint"] = boosted.get("Complaint", 0) * 0.5
        heuristic_flags.append("uncertainty_language")

    # ── Complaint signals ──
    complaint_words = r'\b(terrible|horrible|awful|worst|hate|sucks|broken|unusable|unacceptable|ridiculous|disgusting|pathetic|waste|scam|fraud|rip.?off|garbage|trash|useless|disappointed|frustrating|infuriating)\b'
    if re.search(complaint_words, lower):
        boosted["Complaint"] = boosted.get("Complaint", 0) + 0.2
        heuristic_flags.append("complaint_keywords")
    if sentiment == "Negative" and tox_score > 0.3:
        boosted["Complaint"] = boosted.get("Complaint", 0) + 0.15
        heuristic_flags.append("negative_toxicity_boost")

    superlative_complaint = r'\b(stupidest|dumbest|worst|most annoying|most useless|most pathetic|most ridiculous)\b.*\b(ever|in my life|of all time|i have ever|i\'ve ever)\b'
    if re.search(superlative_complaint, lower):
        boosted["Complaint"] = boosted.get("Complaint", 0) + 0.4
        heuristic_flags.append("superlative_complaint")

    # ── Praise signals ──
    praise_words = r'\b(amazing|awesome|excellent|fantastic|wonderful|brilliant|outstanding|perfect|incredible|love|loved|loving|best|great|superb|magnificent|phenomenal|thank|thanks|grateful|impressed)\b'
    if re.search(praise_words, lower):
        boosted["Praise"] = boosted.get("Praise", 0) + 0.2
        heuristic_flags.append("praise_keywords")
    if sentiment == "Positive":
        boosted["Praise"] = boosted.get("Praise", 0) + 0.1
        heuristic_flags.append("positive_sentiment_boost")

    slang_praise_words = r'\b(fire|goat|lit|sick|bussin|bussing|slaps|slap|dope|heat|chef\'s kiss|elite|insane|valid|hard|goes hard|hits different|peak|god.?tier|next level|top.?tier|no cap|on point|clean|mint|ace|bangin|banging|killer|banger)\b'
    if re.search(slang_praise_words, lower):
        boosted["Praise"] = boosted.get("Praise", 0) + 0.35
        boosted["Complaint"] = boosted.get("Complaint", 0) * 0.4
        heuristic_flags.append("slang_praise")

    # ── Feedback signals ──
    feedback_words = r'\b(should|could|would be better|suggest|suggestion|recommend|consider|improve|improvement|perhaps|it would be nice|feature request|adding|please add|wish|hope)\b'
    if re.search(feedback_words, lower):
        boosted["Feedback"] = boosted.get("Feedback", 0) + 0.2
        heuristic_flags.append("feedback_keywords")

    # ── Spam signals (multiplicative) ──
    spam_multiplier = 1.0

    spam_keywords = r'\b(buy now|click here|free|discount|offer|promo|subscribe|check out my|follow me|visit|earn money|make money|limited time|act now|order now|deal|sale|giveaway|giveaways|win a|winner|signup|sign up)\b'
    spam_keyword_matches = len(re.findall(spam_keywords, lower))
    if spam_keyword_matches >= 1:
        spam_multiplier *= (1.0 + 0.4 * spam_keyword_matches)
        heuristic_flags.append("spam_keywords")

    if re.search(r'(www\.|\.com|\.net|\.org|\.io|https?://)', lower):
        spam_multiplier *= 1.8
        heuristic_flags.append("spam_url")

    exclamation_count = len(re.findall(r'!', text))
    if exclamation_count >= 2:
        spam_multiplier *= (1.0 + 0.2 * min(exclamation_count, 5))
        heuristic_flags.append("spam_exclamations")

    caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
    if len(caps_words) >= 1:
        spam_multiplier *= (1.0 + 0.3 * min(len(caps_words), 5))
        heuristic_flags.append("spam_caps")

    imperative_cmds = r'(?:^|[.!?]\s*)(buy|shop|click|visit|follow|subscribe|join|sign up|check out|order|call|get your|grab|hurry|act now|don\'t miss)\b'
    if re.search(imperative_cmds, lower):
        spam_multiplier *= 1.5
        heuristic_flags.append("spam_imperative")

    if spam_multiplier > 1.0:
        base_spam = max(boosted.get("Spam", 0), 0.08)
        boosted["Spam"] = base_spam * spam_multiplier
        if spam_multiplier >= 2.5:
            boosted["Praise"] = boosted.get("Praise", 0) * 0.3

    # ── Normalize ──
    total = sum(boosted.values())
    if total > 0:
        boosted = {k: round(v / total, 4) for k, v in boosted.items()}

    return boosted, heuristic_flags

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

    Uses contrast-weighted aggregation: sentences containing contrast/concession
    markers (but, however, although, despite, etc.) receive 2x weight because
    they typically contain the author's actual conclusion/stance after introducing
    a concession. For example:
      "The UI is beautiful, but the performance is terrible."
      → "terrible" clause carries more weight than simple averaging.
    """
    sentences = split_sentences(cleaned)

    # If single sentence or very short, skip aggregation
    if len(sentences) <= 1:
        return None

    per_sentence = []
    agg_scores = {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}
    has_contrast = False
    total_weight = 0.0

    for sent_text in sentences:
        # Contrast marker in this sentence → give it more weight
        sentence_has_contrast = bool(_CONTRAST_RE.search(sent_text))
        if sentence_has_contrast:
            has_contrast = True
        weight = 2.0 if sentence_has_contrast else 1.0

        try:
            sentence_input, _ = truncate_for_model(sent_text, "sentiment")
            result = sentiment_classifier(sentence_input)
            scores = _normalize_sentiment_output(result)

            predicted = max(scores, key=scores.get)
            per_sentence.append({
                "text": sent_text,
                "sentiment": predicted,
                "scores": scores,
                "weight": weight,
                "has_contrast_marker": sentence_has_contrast,
            })

            for k in agg_scores:
                agg_scores[k] += scores.get(k, 0) * weight
            total_weight += weight
        except Exception as exc:
            LOGGER.warning("Sentence-level classification failed for %r: %s", sent_text[:60], exc)
            per_sentence.append({
                "text": sent_text,
                "sentiment": "Neutral",
                "scores": {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0},
                "weight": weight,
                "has_contrast_marker": sentence_has_contrast,
            })
            agg_scores["Neutral"] += 1.0 * weight
            total_weight += weight

    if total_weight > 0:
        agg_scores = {k: round(v / total_weight, 4) for k, v in agg_scores.items()}

    return {
        "sentences": per_sentence,
        "aggregated_scores": agg_scores,
        "aggregated_sentiment": max(agg_scores, key=agg_scores.get),
        "is_mixed": len(set(s["sentiment"] for s in per_sentence)) > 1,
        "contrast_weighted": has_contrast,
    }

def analyze_word_sentiment(text: str) -> tuple[list, dict]:
    tokens = re.findall(r'\S+|\s+', text)
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

        context_start = max(0, word_idx - 2)
        context_words = word_positions[context_start:word_idx + 1]
        context_phrase = ' '.join(context_words)
        if len(context_words) > 1:
            phrase_score = vader_analyzer.polarity_scores(context_phrase)['compound']
            single_score = vader_analyzer.polarity_scores(token)['compound']
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

def _ensure_core_models():
    missing = [name for name in ("sentiment", "toxicity", "type") if model_registry.get(name) is None]
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"Required models not loaded: {', '.join(missing)}. Please try again later.",
        )

def _build_text_record(text: str, context: str = "") -> dict:
    cleaned = preprocess_text(text)
    cleaned_context = preprocess_text(context) if context else ""
    # Build context-aware model input: context is prepended so the transformer
    # can attend across both context and text during classification.
    contextual_input = build_contextual_input(cleaned, cleaned_context)
    record = {
        "text": text,
        "context": context,
        "cleaned": cleaned,
        "contextual_input": contextual_input,
        "gibberish": is_gibberish(text),
        "is_english": detect_language_is_english(text),
        "truncation": {},
        "stage_timings_ms": {},
        "heuristics_applied": [],
        "context_signals": detect_context_signals(text),
    }
    for model_name in MODEL_SPECS:
        prepared, meta = truncate_for_model(contextual_input, model_name)
        record[f"{model_name}_input"] = prepared
        record["truncation"][model_name] = meta
    return record

def _apply_sarcasm_adjustment(record: dict, predicted_sentiment: str, sent_scores: dict, sarcasm_score: float) -> tuple[str, dict, bool]:
    lower_text = record["text"].lower()
    has_negative_cues = bool(re.search(
        r'\b(crash|broke|broken|fail|problem|issue|wrong|bad|worst|terrible|horrible|sucks|hate)\b'
        r'|(\d+\s*times)',
        lower_text
    ))
    has_gratitude = bool(re.search(
        r'\b(thanks?|thx|ty|tysm|thank you|grateful|appreciate|thats? great|good job|nice work|well done|awesome work)\b',
        lower_text
    ))
    if has_gratitude:
        has_negative_cues = False
        record["heuristics_applied"].append("sarcasm_gratitude_suppression")

    is_sarcastic = sarcasm_score > 0.80 and has_negative_cues
    if is_sarcastic:
        record["heuristics_applied"].append("sarcasm_negative_context")

    if is_sarcastic and predicted_sentiment == "Positive":
        record["heuristics_applied"].append("sarcasm_flip_positive_to_negative")
        predicted_sentiment = "Negative"
        old_pos = sent_scores.get("Positive", 0)
        old_neg = sent_scores.get("Negative", 0)
        sent_scores["Positive"] = round(old_neg * 0.5, 4)
        sent_scores["Negative"] = round(old_pos * 0.8, 4)
        sent_scores["Neutral"] = round(max(0.0, 1.0 - sent_scores["Positive"] - sent_scores["Negative"]), 4)
    return predicted_sentiment, sent_scores, is_sarcastic

def classify_texts_internal(texts: list[str], contexts: list[str] = None) -> list[dict]:
    _ensure_core_models()
    started = time.perf_counter()
    if contexts is None:
        contexts = [""] * len(texts)
    records = [_build_text_record(str(text), ctx) for text, ctx in zip(texts, contexts)]

    sentiment_results, sentiment_ms = _run_stage_batch(
        "sentiment",
        [record["sentiment_input"] for record in records],
    )
    for record, result in zip(records, sentiment_results):
        record["stage_timings_ms"]["sentiment"] = sentiment_ms
        record["sent_scores"] = _normalize_sentiment_output(result)
        record["predicted_sentiment"] = max(record["sent_scores"], key=record["sent_scores"].get)

    sarcasm_results, sarcasm_ms = _run_stage_batch(
        "sarcasm",
        [record["sarcasm_input"] for record in records],
    )
    for record, result in zip(records, sarcasm_results):
        record["stage_timings_ms"]["sarcasm"] = sarcasm_ms
        sarcasm_score = 0.0 if result is None else _extract_label_score(result, {"irony", "label_1", "1"})
        record["sarcasm_score"] = sarcasm_score
        adjusted_sentiment, adjusted_scores, is_sarcastic = _apply_sarcasm_adjustment(
            record, record["predicted_sentiment"], dict(record["sent_scores"]), sarcasm_score
        )
        record["predicted_sentiment"] = adjusted_sentiment
        record["sent_scores"] = adjusted_scores
        record["is_sarcastic"] = is_sarcastic

    toxicity_results, toxicity_ms = _run_stage_batch(
        "toxicity",
        [record["toxicity_input"] for record in records],
    )
    for record, result in zip(records, toxicity_results):
        record["stage_timings_ms"]["toxicity"] = toxicity_ms
        tox_score = 0.0 if result is None else _extract_label_score(result, {"toxic"})
        record["toxicity"] = tox_score
        record["is_toxic"] = tox_score > 0.5

    emotion_results, emotion_ms = _run_stage_batch(
        "emotion",
        [record["emotion_input"] for record in records],
    )
    for record, result in zip(records, emotion_results):
        record["stage_timings_ms"]["emotion"] = emotion_ms
        if result is None:
            record["emotions"] = []
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            record["emotions"] = result
        elif isinstance(result, list) and result:
            record["emotions"] = result[0]
        else:
            record["emotions"] = []

    type_results, type_ms = _run_stage_batch(
        "type",
        [record["type_input"] for record in records],
        candidate_labels=CANDIDATE_TYPES,
    )
    for record, result in zip(records, type_results):
        record["stage_timings_ms"]["type"] = type_ms
        raw_type_scores = {}
        for label, score in zip(result["labels"], result["scores"]):
            clean_label = TYPE_LABEL_MAP.get(label, label)
            raw_type_scores[clean_label] = round(score, 4)

        type_scores, heuristic_flags = apply_type_heuristics(
            record["text"],
            raw_type_scores,
            record["predicted_sentiment"],
            record["toxicity"],
            record["emotions"],
        )
        record["heuristics_applied"].extend(heuristic_flags)

        if record["is_sarcastic"] and record["predicted_sentiment"] == "Negative":
            praise_val = type_scores.get("Praise", 0)
            type_scores["Complaint"] = type_scores.get("Complaint", 0) + praise_val * 0.6
            type_scores["Praise"] = praise_val * 0.3
            total_t = sum(type_scores.values())
            if total_t > 0:
                type_scores = {k: round(v / total_t, 4) for k, v in type_scores.items()}
            record["heuristics_applied"].append("sarcasm_type_redirect")

        record["type_scores"] = type_scores
        record["predicted_type"] = max(type_scores, key=type_scores.get)
        record["predicted_sentiment"], record["is_uncertain"] = apply_confidence_flag(
            record["predicted_sentiment"], record["sent_scores"]
        )

    responses = []
    total_latency_ms = round((time.perf_counter() - started) * 1000)
    per_item_latency_ms = round(total_latency_ms / max(1, len(records)))

    for record in records:
        multi_sentence = classify_multi_sentence(record["text"], record["cleaned"])
        if multi_sentence and multi_sentence["is_mixed"]:
            record["sent_scores"] = multi_sentence["aggregated_scores"]
            record["predicted_sentiment"] = multi_sentence["aggregated_sentiment"]
            record["predicted_sentiment"], record["is_uncertain"] = apply_confidence_flag(
                record["predicted_sentiment"], record["sent_scores"]
            )
            record["heuristics_applied"].append("mixed_sentiment_aggregation")

        if record["gibberish"]:
            record["predicted_sentiment"] = "Neutral"
            record["sent_scores"] = {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0}
            record["predicted_type"] = "Spam"
            record["type_scores"] = {label: (1.0 if label == "Spam" else 0.0) for label in TYPE_LABEL_MAP.values()}
            record["is_uncertain"] = False
            record["emotions"] = []
            record["is_sarcastic"] = False
            record["sarcasm_score"] = 0.0
            record["heuristics_applied"].append("gibberish_short_circuit")

        # ── Nuclear enrichment pipeline ────────────────────────────────────────
        # 1. Rhetorical question adjustment (before VADER calibration)
        record["predicted_sentiment"], record["sent_scores"] = apply_rhetorical_question_adjustment(
            record["text"], record["predicted_sentiment"], record["sent_scores"],
            record["heuristics_applied"],
        )

        # 2. VADER cross-calibration (especially useful for short/ambiguous texts)
        record["predicted_sentiment"], record["sent_scores"] = calibrate_with_vader(
            record["text"], record["predicted_sentiment"], record["sent_scores"],
            record["heuristics_applied"],
        )

        # 3. Re-evaluate confidence after calibration
        record["predicted_sentiment"], record["is_uncertain"] = apply_confidence_flag(
            record["predicted_sentiment"], record["sent_scores"]
        )

        # 4. Sensitive topic detection — does NOT change sentiment, adds context
        record["sensitive_topics"] = detect_sensitive_topics(record["text"])

        # 5. Subjectivity analysis
        record["subjectivity"] = compute_subjectivity(record["text"])

        # 6. Sentiment intensity
        record["sentiment_intensity"] = compute_sentiment_intensity(
            record["predicted_sentiment"], record["sent_scores"], record["text"]
        )

        # 7. Ambivalence — both positive and negative scores are meaningfully elevated
        pos_conf = record["sent_scores"].get("Positive", 0.0)
        neg_conf = record["sent_scores"].get("Negative", 0.0)
        record["is_ambivalent"] = (pos_conf > 0.28 and neg_conf > 0.28)

        # 8. Text quality metadata
        word_count = len(record["text"].split())
        record["text_quality"] = {
            "word_count": word_count,
            "is_very_short": word_count < 5,
            "short_text_note": (
                "Very short text — classification confidence may be lower."
                if word_count < 5 else None
            ),
        }
        # ── End nuclear enrichment ─────────────────────────────────────────────

        word_analysis, word_counts = analyze_word_sentiment(record["text"])
        response = {
            "sentiment": record["predicted_sentiment"],
            "sentiment_confidence": {
                "positive": record["sent_scores"].get("Positive", 0),
                "neutral": record["sent_scores"].get("Neutral", 0),
                "negative": record["sent_scores"].get("Negative", 0),
            },
            "sentiment_intensity": record["sentiment_intensity"],
            "is_uncertain": record["is_uncertain"],
            "is_ambivalent": record["is_ambivalent"],
            "toxicity": round(record.get("toxicity", 0.0), 4),
            "is_toxic": record.get("is_toxic", False),
            "comment_type": record["predicted_type"],
            "type_scores": record["type_scores"],
            "is_sarcastic": record.get("is_sarcastic", False),
            "sarcasm_score": round(record.get("sarcasm_score", 0.0), 4),
            "emotions": [
                {"label": e["label"], "score": round(e["score"], 4)}
                for e in record.get("emotions", [])[:5]
            ],
            "is_english": record["is_english"],
            # Context-awareness fields
            "context_injected": bool(record.get("context", "")),
            "context_signals": record.get("context_signals", {}),
            # Nuance fields
            "sensitive_topics": record["sensitive_topics"],
            "subjectivity": record["subjectivity"],
            "text_quality": record["text_quality"],
            "word_analysis": word_analysis,
            "word_counts": word_counts,
            "latency_ms": per_item_latency_ms,
            "stage_timings_ms": record["stage_timings_ms"],
            "truncation": record["truncation"],
            "heuristics_applied": sorted(set(record["heuristics_applied"])),
            "model_versions": {
                name: model_status.get(name, {}).get("model")
                for name in MODEL_SPECS
                if model_status.get(name, {}).get("loaded")
            },
        }
        if multi_sentence:
            response["multi_sentence"] = {
                "sentences": multi_sentence["sentences"],
                "is_mixed": multi_sentence["is_mixed"],
                "contrast_weighted": multi_sentence.get("contrast_weighted", False),
            }
        responses.append(response)

    return responses

def classify_text_internal(text: str, context: str = "") -> dict:
    return classify_texts_internal([text], contexts=[context])[0]

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
        try:
            batch_results = classify_texts_internal([str(text) for text in batch])
        except Exception:
            LOGGER.exception("Batch inference failed for rows %s-%s", i, i + len(batch) - 1)
            batch_results = [None] * len(batch)

        for text, result in zip(batch, batch_results):
            try:
                if result is None:
                    raise RuntimeError("No result returned from batch inference")
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
                    "word_counts": result.get("word_counts", {}),
                    "stage_timings_ms": result.get("stage_timings_ms", {}),
                    "heuristics_applied": result.get("heuristics_applied", []),
                    "truncation": result.get("truncation", {}),
                })
            except Exception as e:
                LOGGER.exception("Error on batch item")
                results.append({
                    "comment": str(text),
                    "sentiment": "Error",
                    "conf_pos": 0, "conf_neu": 0, "conf_neg": 0,
                    "toxicity": 0, "is_toxic": False, "comment_type": "Unknown",
                    "is_sarcastic": False, "is_uncertain": False,
                    "emotions": [],
                    "word_analysis": [], "word_counts": {},
                    "stage_timings_ms": {},
                    "heuristics_applied": [f"batch_error:{type(e).__name__}"],
                    "truncation": {},
                })
            job["processed"] = len(results)

    job["status"] = "done"
    job["results"] = results

# ──────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────

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
        "status": "ok" if (required_ready and optional_ready) else ("degraded" if required_ready else "model_not_loaded"),
        "sentiment_loaded": sentiment_classifier is not None,
        "toxicity_loaded": toxicity_classifier is not None,
        "type_loaded": type_classifier is not None,
        "emotion_loaded": emotion_classifier is not None,
        "sarcasm_loaded": sarcasm_classifier is not None,
        "preferred_sentiment_model": _get_configured_modernbert_model() or None,
        "active_sentiment_model": model_status.get("sentiment", {}).get("model"),
        "active_sentiment_display_name": model_status.get("sentiment", {}).get("display_name"),
        "model_status": model_status,
        "version": APP_VERSION
    }

@app.post("/classify/text")
async def classify_text_endpoint(request: Request):
    check_rate_limit(request.client.host)

    body = await request.json()
    text = body.get("text", "").strip()
    # Optional surrounding context: the post/thread/topic being replied to.
    # When provided it is prepended to the model input so the transformer can
    # attend across both context and comment during classification.
    context = body.get("context", "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Please enter a comment. Text field cannot be empty.")
    if len(text) > 8192:
        raise HTTPException(status_code=400, detail="Text exceeds maximum length of 8192 characters.")
    if len(context) > 2048:
        raise HTTPException(status_code=400, detail="Context exceeds maximum length of 2048 characters.")

    result = classify_text_internal(text, context=context)
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
