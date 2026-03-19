import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as backend_main


class FakeTokenizer:
    def __call__(self, text, truncation=False, max_length=None, return_overflowing_tokens=False, return_attention_mask=False):
        tokens = text.split()
        sliced = tokens[:max_length] if truncation and max_length else tokens
        overflow = tokens[max_length:] if truncation and max_length and len(tokens) > max_length else []
        payload = {"input_ids": sliced}
        if return_overflowing_tokens:
            payload["overflowing_tokens"] = overflow
        return payload

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return " ".join(token_ids)


class FakeBatchClassifier:
    def __init__(self, fn):
        self.fn = fn
        self.batch_lengths = []

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.batch_lengths.append(len(texts))
        results = [self.fn(text, **kwargs) for text in texts]
        return results if len(results) > 1 else results[0]


def fake_sentiment(text, **kwargs):
    lower = text.lower()
    if "bad" in lower or "crashed" in lower or "terrible" in lower:
        return [
            {"label": "negative", "score": 0.88},
            {"label": "neutral", "score": 0.08},
            {"label": "positive", "score": 0.04},
        ]
    if "great" in lower or "love" in lower or "amazing" in lower:
        return [
            {"label": "positive", "score": 0.84},
            {"label": "neutral", "score": 0.10},
            {"label": "negative", "score": 0.06},
        ]
    return [
        {"label": "neutral", "score": 0.80},
        {"label": "positive", "score": 0.10},
        {"label": "negative", "score": 0.10},
    ]


def fake_sarcasm(text, **kwargs):
    lower = text.lower()
    if "great" in lower and "crashed" in lower:
        return [
            {"label": "non_irony", "score": 0.09},
            {"label": "irony", "score": 0.91},
        ]
    return [
        {"label": "non_irony", "score": 0.95},
        {"label": "irony", "score": 0.05},
    ]


def fake_toxicity(text, **kwargs):
    score = 0.77 if "idiot" in text.lower() else 0.04
    return [
        {"label": "toxic", "score": score},
        {"label": "non_toxic", "score": 1 - score},
    ]


def fake_emotion(text, **kwargs):
    lower = text.lower()
    if "thanks" in lower or "love" in lower:
        return [{"label": "gratitude", "score": 0.81}]
    if "crashed" in lower:
        return [{"label": "annoyance", "score": 0.74}]
    return [{"label": "neutral", "score": 0.90}]


def fake_type(text, candidate_labels=None, **kwargs):
    lower = text.lower()
    if "?" in lower:
        scores = [0.74, 0.08, 0.06, 0.04, 0.04, 0.04]
    elif "buy now" in lower:
        scores = [0.02, 0.02, 0.03, 0.82, 0.05, 0.06]
    elif "great" in lower or "love" in lower:
        scores = [0.05, 0.05, 0.05, 0.05, 0.76, 0.04]
    else:
        scores = [0.05, 0.10, 0.68, 0.05, 0.06, 0.06]
    return {"labels": candidate_labels, "scores": scores}


class BackendRegressionTests(unittest.TestCase):
    def setUp(self):
        for model_name in backend_main.MODEL_SPECS:
            backend_main.model_tokenizers[model_name] = FakeTokenizer()

        backend_main.sentiment_classifier = FakeBatchClassifier(fake_sentiment)
        backend_main.toxicity_classifier = FakeBatchClassifier(fake_toxicity)
        backend_main.type_classifier = FakeBatchClassifier(fake_type)
        backend_main.emotion_classifier = FakeBatchClassifier(fake_emotion)
        backend_main.sarcasm_classifier = FakeBatchClassifier(fake_sarcasm)

        backend_main.model_registry = {
            "sentiment": backend_main.sentiment_classifier,
            "toxicity": backend_main.toxicity_classifier,
            "type": backend_main.type_classifier,
            "emotion": backend_main.emotion_classifier,
            "sarcasm": backend_main.sarcasm_classifier,
        }
        backend_main.model_status = {
            name: {
                "loaded": True,
                "required": backend_main.MODEL_SPECS[name]["required"],
                "model": backend_main._get_primary_model_identifier(name),
                "display_name": backend_main.MODEL_SPECS[name].get("display_name", backend_main._get_primary_model_identifier(name)),
                "max_tokens": backend_main.MODEL_SPECS[name]["max_tokens"],
                "error": None,
                "attempted_models": [backend_main._get_primary_model_identifier(name)],
            }
            for name in backend_main.MODEL_SPECS
        }
        backend_main.jobs.clear()

    def test_preprocess_text_normalizes_slang_and_hashtags(self):
        text = "OMG #LoveThis sooo much!!!"
        processed = backend_main.preprocess_text(text)
        self.assertIn("oh my god", processed.lower())
        self.assertIn("Love This", processed)
        self.assertNotIn("sooo", processed)

    def test_tokenizer_truncation_is_token_aware(self):
        original_max = backend_main.MODEL_SPECS["sentiment"]["max_tokens"]
        backend_main.MODEL_SPECS["sentiment"]["max_tokens"] = 4
        try:
            prepared, meta = backend_main.truncate_for_model(
                "one two three four five six",
                "sentiment",
            )
        finally:
            backend_main.MODEL_SPECS["sentiment"]["max_tokens"] = original_max
        self.assertEqual(prepared, "one two three four")
        self.assertTrue(meta["truncated"])

    def test_sarcasm_uses_non_top1_irony_score(self):
        result = backend_main.classify_text_internal("Oh great, crashed 5 times today")
        self.assertEqual(result["sentiment"], "Negative")
        self.assertTrue(result["is_sarcastic"])
        self.assertGreater(result["sarcasm_score"], 0.8)
        self.assertIn("sarcasm_negative_context", result["heuristics_applied"])

    def test_batch_inference_uses_real_batched_model_calls(self):
        results = backend_main.classify_texts_internal([
            "I love this product",
            "This app crashed again",
        ])
        self.assertEqual(len(results), 2)
        self.assertIn(2, backend_main.sentiment_classifier.batch_lengths)
        self.assertIn(2, backend_main.type_classifier.batch_lengths)

    def test_health_endpoint_exposes_model_status(self):
        backend_main.load_model = lambda: None
        with TestClient(backend_main.app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("model_status", payload)
        self.assertIn("active_sentiment_model", payload)

    def test_sentiment_model_candidates_include_fallback(self):
        candidates = backend_main._resolve_model_candidates("sentiment")
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[-1]["model"], backend_main.DEFAULT_SENTIMENT_FALLBACK_MODEL)


if __name__ == "__main__":
    unittest.main()
