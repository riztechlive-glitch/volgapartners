"""
Post-processor — cleans, segments, and enriches raw transcripts.

Design decisions:
- Pure functions where possible (easy to unit test).
- Keyword extraction uses TF-IDF via scikit-learn — no external API needed.
- Falls back to frequency-based extraction if scikit-learn is unavailable.
"""

from __future__ import annotations

import logging
import re
import string
from collections import Counter

from .config import PipelineConfig
from .models import ProcessedTranscript, RawTranscript

logger = logging.getLogger(__name__)

# ── Cleaning ──────────────────────────────────────────────────────────────────

_FILLER_WORDS = frozenset({
    "um", "uh", "like", "you know", "so", "actually", "basically",
    "i mean", "right", "well", "kind of", "sort of",
})

# Common disfluency patterns (hesitations, repetitions)
_DISFLUENCY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(um|uh|erm|ah)\b", re.IGNORECASE), ""),
    (re.compile(r"\s{2,}"), " "),           # collapse whitespace
    (re.compile(r"(.)\1{2,}"), r"\1\1"),    # "weeell" → "weel" (keep doubles)
]


def clean_text(text: str) -> str:
    """Remove fillers, normalize whitespace, fix punctuation spacing."""
    result = text
    for pattern, replacement in _DISFLUENCY_PATTERNS:
        result = pattern.sub(replacement, result)

    # Normalize punctuation spacing: "word , word" → "word, word"
    result = re.sub(r"\s([,.!?;:])", r"\1", result)

    return result.strip()


# ── Sentence Segmentation ─────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Uses a simple heuristic (periods, question marks, exclamation marks)
    followed by a space and an uppercase letter. This avoids pulling in
    a heavyweight NLP library for a basic need.
    """
    # Split on sentence-ending punctuation followed by whitespace + uppercase
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in raw if s.strip()]


# ── Keyword Extraction ────────────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with "
    "at by from as into through during before after above below between "
    "out off over under again further then once here there when where "
    "why how all both each every few more most other some such no not "
    "only own same so than too very it its i me my we our you your "
    "he him his she her they them their this that these those and but "
    "if or because while although since unless until about".split()
)


def extract_keywords_frequency(text: str, max_keywords: int = 10) -> list[str]:
    """TF-IDF-free keyword extraction using word frequency + stop-word removal."""
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(max_keywords)]


def extract_keywords_tfidf(texts: list[str], max_keywords: int = 10) -> list[str]:
    """TF-IDF keyword extraction — requires scikit-learn."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            max_features=max_keywords * 2,
            stop_words="english",
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Average TF-IDF across all documents
        scores = tfidf_matrix.mean(axis=0).A1
        top_indices = scores.argsort()[::-1][:max_keywords]

        return [feature_names[i] for i in top_indices if scores[i] > 0]
    except ImportError:
        logger.warning("scikit-learn not installed; falling back to frequency-based keywords.")
        return extract_keywords_frequency(texts[0], max_keywords)


# ── Main Processor ────────────────────────────────────────────────────────────

def process_transcript(raw: RawTranscript, config: PipelineConfig) -> ProcessedTranscript:
    """
    Full post-processing pipeline:
    1. Clean text (remove fillers, normalize)
    2. Split into sentences
    3. Extract keywords
    4. Compute metadata (word count, speaking rate)
    """
    logger.info("Processing transcript (%d segments, %.1fs)...",
                len(raw.segments), raw.duration_seconds)

    cleaned = clean_text(raw.full_text)
    sentences = split_sentences(cleaned)

    # Keyword extraction — use TF-IDF across segments for better results
    if config.extract_keywords:
        segment_texts = [seg.text for seg in raw.segments]
        keywords = extract_keywords_tfidf(segment_texts, config.max_keywords)
    else:
        keywords = []

    word_count = len(cleaned.split())
    duration_minutes = max(raw.duration_seconds / 60, 0.01)
    speaking_rate = word_count / duration_minutes

    result = ProcessedTranscript(
        raw=raw,
        cleaned_text=cleaned,
        sentences=sentences,
        keywords=keywords,
        word_count=word_count,
        speaking_rate_wpm=speaking_rate,
    )

    logger.info("Processing complete: %d words, %d sentences, %d keywords.",
                result.word_count, len(result.sentences), len(result.keywords))

    return result
