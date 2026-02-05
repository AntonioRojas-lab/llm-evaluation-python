"""
src/run_eval.py

Offline evaluation runner (no API keys required).

What a reviewer should be able to do:
- run this script
- see a clear console summary
- get a CSV report with per-example scores + aggregate metrics

This script:
1) Loads eval_set.csv
2) Generates predictions in "mock" mode (deterministic, no network)
3) Scores predictions using src/metrics.py
4) Writes reports/latest_results.csv and reports/summary.json
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

from metrics import aggregate_mean, default_metric_registry, score_prediction, refusal_rate, banned_terms_hit


# -------------------------
# Config (simple on purpose)
# -------------------------

DEFAULT_EVAL_PATH = os.path.join("data", "eval_set.csv")
DEFAULT_SYSTEM_PROMPT_PATH = os.path.join("prompts", "system_prompt.txt")
REPORTS_DIR = "reports"

# Enterprise-style checks (example). Keep small and obvious.
DEFAULT_BANNED_TERMS = [
    "password",
    "ssn",
]


# -------------------------
# Data loading
# -------------------------

def load_text_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval set not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def pick_columns(fieldnames: List[str]) -> Tuple[str, str, str]:
    """
    Tries to auto-detect columns so the repo is easy to run.

    Returns: (id_col, input_col, expected_col)
    """
    def first_match(candidates: List[str]) -> str:
        for c in candidates:
            if c in fieldnames:
                return c
        return ""

    id_col = first_match(["id", "example_id", "uid"])
    input_col = first_match(["input", "prompt", "question", "query", "user_input"])
    expected_col = first_match(["expected", "gold", "reference", "target", "answer"])

    if not input_col or not expected_col:
        raise ValueError(
            "Could not detect required columns.\n"
            f"Found columns: {fieldnames}\n"
            "Expected something like: input/prompt/question AND expected/gold/reference."
        )

    # If there's no id column, we’ll generate one.
    if not id_col:
        id_col = "__generated_id__"

    return id_col, input_col, expected_col


# -------------------------
# Mock model (deterministic)
# -------------------------

class MockLLM:
    """
    Deterministic mock generator.
    The goal is not "accuracy" — it's to let anyone run the evaluation pipeline offline.

    Strategy:
    - If input looks like a simple factual question we can answer via tiny lookup, answer it.
    - Otherwise, produce a short structured answer.
    - Sometimes refuse for certain triggers to exercise refusal_rate.
    """

    def __init__(self, system_prompt: str = "") -> None:
        self.system_prompt = system_prompt

        self._capital_lookup = {
            "france": "Paris",
            "spain": "Madrid",
            "italy": "Rome",
            "germany": "Berlin",
            "portugal": "Lisbon",
            "mexico": "Mexico City",
            "argentina": "Buenos Aires",
            "chile": "Santiago",
            "peru": "Lima",
            "colombia": "Bogotá",
        }

        self._refusal_triggers = [
            "hack",
            "illegal",
            "steal",
            "password",
            "ssn",
        ]

    def generate(self, user_input: str) -> str:
        text = (user_input or "").strip()
        low = text.lower()

        # trigger a refusal in a predictable way
        if any(t in low for t in self._refusal_triggers):
            return "Lo siento, no puedo ayudar con eso."

        # very small "factual" behavior: capitals
        if "capital" in low:
            # naive country extraction: check if any known country name appears
            for country, cap in self._capital_lookup.items():
                if country in low:
                    return cap

        # generic response
        return (
            "Respuesta:\n"
            "- Idea principal: " + (text[:80] + ("..." if len(text) > 80 else "")) + "\n"
            "- Nota: salida mock (sin modelo externo)."
        )


# -------------------------
# Reporting
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_summary(agg: Dict[str, float], n_rows: int) -> None:
    print("\n=== Evaluation Summary ===")
    print(f"Examples: {n_rows}")
    for k in sorted(agg.keys()):
        print(f"{k:>16}: {agg[k]:.3f}")
    print("==========================\n")


# -------------------------
# Main
# -------------------------

def main() -> None:
    eval_path = DEFAULT_EVAL_PATH
    rows, fieldnames = read_csv(eval_path)
    id_col, input_col, expected_col = pick_columns(fieldnames)

    system_prompt = load_text_file(DEFAULT_SYSTEM_PROMPT_PATH)
    model = MockLLM(system_prompt=system_prompt)

    metric_registry = default_metric_registry()
    scored_rows: List[Dict[str, str]] = []
    per_row_metric_dicts: List[Dict[str, float]] = []

    # If we
