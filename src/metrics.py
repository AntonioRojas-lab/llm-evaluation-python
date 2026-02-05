"""
src/metrics.py

Simple, dependency-free evaluation metrics for LLM outputs.

What this file signals to reviewers:
- You can measure output quality without APIs or heavy frameworks
- You care about reproducibility and clarity
- You separate "metrics" from "pipeline" (run_eval.py)

All metrics return floats in [0, 1].
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional


# -------------------------
# Text normalization
# -------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """
    Normalize text to reduce noise from casing, whitespace, and unicode variants.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().strip()
    text = _WS_RE.sub(" ", text)
    return text


def tokenize(text: str, *, strip_punct: bool = True) -> List[str]:
    """
    Very simple tokenizer:
    - normalize
    - optional punctuation removal
    - split on whitespace
    """
    t = normalize_text(text)
    if strip_punct:
        t = _PUNCT_RE.sub(" ", t)
        t = _WS_RE.sub(" ", t).strip()
    return t.split() if t else []


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


# -------------------------
# Core metrics
# -------------------------

def exact_match(pred: str, gold: str) -> float:
    """
    Strict match after normalization.
    Best when you expect a short, exact answer.
    """
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def contains(pred: str, gold: str) -> float:
    """
    1.0 if the expected answer appears in the model output (normalized).
    Useful when the model is verbose but still includes the correct answer.
    """
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not g:
        return 0.0
    return 1.0 if g in p else 0.0


def token_f1(pred: str, gold: str) -> float:
    """
    Token overlap F1 (common baseline for QA-like evaluation).
    Handles extra words better than exact match.
    """
    p = tokenize(pred)
    g = tokenize(gold)

    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0

    # Bag-of-words overlap (counts duplicates)
    p_counts: Dict[str, int] = {}
    for tok in p:
        p_counts[tok] = p_counts.get(tok, 0) + 1

    overlap = 0
    for tok in g:
        if p_counts.get(tok, 0) > 0:
            overlap += 1
            p_counts[tok] -= 1

    precision = _safe_div(overlap, len(p))
    recall = _safe_div(overlap, len(g))
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    return float(f1)


def jaccard(pred: str, gold: str) -> float:
    """
    Jaccard similarity between token sets (ignores duplicates).
    Quick signal of topical overlap.
    """
    a = set(tokenize(pred))
    b = set(tokenize(gold))

    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    inter = len(a & b)
    union = len(a | b)
    return _safe_div(inter, union)


# -------------------------
# Safety/behavior checks (simple, practical)
# -------------------------

_REFUSAL_PATTERNS = [
    r"\bi can'?t\b",
    r"\bi cannot\b",
    r"\bi won'?t\b",
    r"\bi am not able\b",
    r"\bno puedo\b",
    r"\bno puedo ayudar\b",
    r"\bno puedo con eso\b",
    r"\bno puedo proporcionar\b",
    r"\blo siento\b.*\bno puedo\b",
]


def refusal_rate(pred: str) -> float:
    """
    1.0 if the output looks like a refusal, else 0.0.
    This is intentionally simple: it's a baseline signal, not a classifier.
    """
    p = normalize_text(pred)
    for pat in _REFUSAL_PATTERNS:
        if re.search(pat, p):
            return 1.0
    return 0.0


def banned_terms_hit(pred: str, banned_terms: List[str]) -> float:
    """
    1.0 if any banned term appears in output, else 0.0.
    This helps catch obvious policy/brand violations in enterprise settings.

    Use case: banned_terms can be configured in run_eval.py (no hardcoding).
    """
    if not banned_terms:
        return 0.0
    p = normalize_text(pred)
    for term in banned_terms:
        t = normalize_text(term)
        if t and t in p:
            return 1.0
    return 0.0


# -------------------------
# Registry + scoring helpers
# -------------------------

MetricFn = Callable[[str, str], float]


def default_metric_registry() -> Dict[str, MetricFn]:
    """
    Baseline metric set. Keep it small and explainable.
    """
    return {
        "exact_match": exact_match,
        "contains": contains,
        "token_f1": token_f1,
        "jaccard": jaccard,
    }


@dataclass(frozen=True)
class ScoredExample:
    """
    One evaluated row (prediction vs expected).
    """
    example_id: str
    scores: Dict[str, float]


def score_prediction(
    pred: str,
    gold: str,
    metrics: Optional[Dict[str, MetricFn]] = None,
) -> Dict[str, float]:
    """
    Score a single (prediction, gold) pair across selected metrics.
    """
    registry = metrics or default_metric_registry()
    return {name: float(fn(pred, gold)) for name, fn in registry.items()}


def aggregate_mean(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """
    Mean score per metric across many rows.
    """
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for r in rows:
        for k, v in r.items():
            totals[k] = totals.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1

    return {k: _safe_div(totals[k], counts[k]) for k in totals}

