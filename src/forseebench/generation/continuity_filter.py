"""Continuity and relevance signals for adaptive context selection."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from math import sqrt
import re
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_LOCATION_CUES = {
    "door",
    "hall",
    "hallway",
    "room",
    "street",
    "kitchen",
    "table",
    "car",
    "bed",
    "stairs",
    "office",
    "bar",
    "restaurant",
}
_ACTION_LEXICON = {
    "open",
    "close",
    "walk",
    "run",
    "sit",
    "stand",
    "turn",
    "look",
    "pick",
    "hold",
    "raise",
    "unlock",
    "enter",
    "leave",
    "hand",
    "give",
    "take",
    "search",
    "pull",
    "push",
    "step",
    "approach",
    "say",
    "ask",
    "smile",
    "stop",
    "drink",
    "eat",
    "drive",
    "knock",
    "call",
}


def safe_float(value: str | None) -> float | None:
    """Convert a timestamp-like string to float when possible."""

    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compute_timestamp_gap(context: list[dict[str, Any]], target: dict[str, Any]) -> float | None:
    """Compute the weak temporal gap signal between context end and target start."""

    if not context:
        return None
    last_end = safe_float(context[-1].get("timestamp_end"))
    target_start = safe_float(target.get("timestamp_start"))
    if last_end is None or target_start is None:
        return None
    return target_start - last_end


def normalize_tokens(text: str) -> list[str]:
    """Tokenize text into simple lowercase content tokens."""

    return [token.lower() for token in _TOKEN_RE.findall(text)]


def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute a lightweight TF-IDF cosine similarity without external deps."""

    tokens_a = [token for token in normalize_tokens(text_a) if token not in _STOPWORDS]
    tokens_b = [token for token in normalize_tokens(text_b) if token not in _STOPWORDS]
    if not tokens_a or not tokens_b:
        return 0.0
    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    vocab = sorted(set(counts_a) | set(counts_b))
    idf: dict[str, float] = {}
    for token in vocab:
        docs = int(token in counts_a) + int(token in counts_b)
        idf[token] = 1.0 + (0.0 if docs == 2 else 0.69314718056)
    vec_a = [counts_a.get(token, 0) * idf[token] for token in vocab]
    vec_b = [counts_b.get(token, 0) * idf[token] for token in vocab]
    dot = sum(left * right for left, right in zip(vec_a, vec_b))
    norm_a = sqrt(sum(value * value for value in vec_a))
    norm_b = sqrt(sum(value * value for value in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def sentence_transformer_similarity(text_a: str, text_b: str) -> float | None:
    """Try sentence-transformers, returning None when unavailable."""

    model = _load_sentence_transformer_model()
    if model is None:
        return None
    from sentence_transformers.util import cos_sim
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    return float(cos_sim(embeddings[0], embeddings[1]).item())


def compute_semantic_similarity(text_a: str, text_b: str) -> tuple[float, str]:
    """Compute semantic similarity with sentence-transformers or TF-IDF fallback."""

    transformer_score = sentence_transformer_similarity(text_a, text_b)
    if transformer_score is not None:
        return transformer_score, "sentence_transformers"
    return tfidf_cosine_similarity(text_a, text_b), "tfidf_fallback"


def _extract_with_spacy(text: str) -> dict[str, list[str]] | None:
    nlp = _load_spacy_model()
    if nlp is None:
        return None
    doc = nlp(text)
    people = sorted({ent.text for ent in doc.ents if ent.label_ == "PERSON"})
    locations = sorted({ent.text for ent in doc.ents if ent.label_ in {"GPE", "LOC", "FAC"}})
    objects = sorted(
        {
            token.lemma_.lower()
            for token in doc
            if token.pos_ in {"NOUN", "PROPN"} and token.lemma_.lower() not in _STOPWORDS
        }
    )
    actions = sorted(
        {
            token.lemma_.lower()
            for token in doc
            if token.pos_ == "VERB" and token.lemma_.lower() not in _STOPWORDS
        }
    )
    return {
        "people": people,
        "objects": objects,
        "locations": locations,
        "actions": actions,
        "extractor": ["spacy"],
    }


def _extract_with_regex(text: str) -> dict[str, list[str]]:
    tokens = normalize_tokens(text)
    people = sorted({token for token in tokens if token.startswith("person") or token in {"man", "woman", "he", "she", "they", "someone", "waiter", "couple"}})
    locations = sorted({token for token in tokens if token in _LOCATION_CUES})
    actions = sorted({token for token in tokens if token in _ACTION_LEXICON or token.endswith("ing") or token.endswith("ed")})
    objects = sorted(
        {
            token
            for token in tokens
            if token not in _STOPWORDS and token not in people and token not in locations and token not in actions
        }
    )
    return {
        "people": people,
        "objects": objects,
        "locations": locations,
        "actions": actions,
        "extractor": ["regex_fallback"],
    }


def extract_entities_actions(text: str) -> dict[str, list[str]]:
    """Extract lightweight continuity entities and actions."""

    return _extract_with_spacy(text) or _extract_with_regex(text)


@lru_cache(maxsize=1)
def _load_sentence_transformer_model() -> Any | None:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _load_spacy_model() -> Any | None:
    try:
        import spacy
    except Exception:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def overlap_ratio(left: list[str], right: list[str]) -> float:
    """Return overlap over target-side unique items."""

    right_set = set(right)
    if not right_set:
        return 0.0
    left_set = set(left)
    return len(left_set & right_set) / len(right_set)


def build_continuity_features(context: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, Any]:
    """Compute continuity features for one context-target pair."""

    context_text = " ".join(row.get("audio_description", "") for row in context)
    target_text = target.get("audio_description", "")
    last_text = context[-1].get("audio_description", "") if context else ""
    sim_last, similarity_backend = compute_semantic_similarity(last_text, target_text)
    sim_mean, _ = compute_semantic_similarity(context_text, target_text)
    context_signals = extract_entities_actions(context_text)
    target_signals = extract_entities_actions(target_text)
    entity_overlap = overlap_ratio(
        sorted(set(context_signals["people"]) | set(context_signals["objects"])),
        sorted(set(target_signals["people"]) | set(target_signals["objects"])),
    )
    action_overlap = overlap_ratio(context_signals["actions"], target_signals["actions"])
    location_overlap = overlap_ratio(context_signals["locations"], target_signals["locations"])
    return {
        "timestamp_gap": compute_timestamp_gap(context, target),
        "semantic_similarity_last": sim_last,
        "semantic_similarity_mean": sim_mean,
        "similarity_backend": similarity_backend,
        "context_entities": context_signals,
        "target_entities": target_signals,
        "entity_overlap": entity_overlap,
        "action_overlap": action_overlap,
        "location_overlap": location_overlap,
        "continuity_scores": {
            "semantic_similarity_last": sim_last,
            "semantic_similarity_mean": sim_mean,
            "entity_overlap": entity_overlap,
            "action_overlap": action_overlap,
            "location_overlap": location_overlap,
        },
    }
