"""
backend/filler_cleaner.py

Removes short, high-frequency filler words from extracted text before
it is embedded and stored in ChromaDB.

Scope: indexing pipeline only. Original files on disk are NEVER modified.
"""

import re
from collections import Counter

import config


# Extensions where short tokens carry heavy semantic weight.
# Cleaning is skipped entirely for these file types.
SKIP_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".cpp", ".c", ".h", ".java", ".cs",
    ".json", ".yaml", ".yml", ".toml", ".xml",
}


def remove_fillers(
    text: str,
    threshold: float = None,
    min_length: int = None,
    file_extension: str = None,
) -> str:
    """
    Removes short, high-frequency filler words from text before indexing.

    Args:
        text:           Raw extracted text from parser.py
        threshold:      Frequency ratio above which a short word is removed.
                        Defaults to config.FILLER_FREQUENCY_THRESHOLD (0.03)
        min_length:     Words strictly shorter than this are candidates.
                        Defaults to config.FILLER_MIN_WORD_LENGTH (3)
        file_extension: Lowercase file extension including dot (e.g. ".py").
                        If the extension is in SKIP_EXTENSIONS, text is
                        returned unchanged.

    Returns:
        Cleaned text string with filler words removed, or the original
        text unchanged if cleaning is disabled / skipped / unnecessary.
    """

    # ------------------------------------------------------------------
    # Safety gates
    # ------------------------------------------------------------------

    if not getattr(config, "ENABLE_FILLER_CLEANING", True):
        return text

    if not text or not text.strip():
        return text

    if file_extension and file_extension.lower() in SKIP_EXTENSIONS:
        return text

    threshold = (
        threshold if threshold is not None
        else config.FILLER_FREQUENCY_THRESHOLD
    )

    min_length = (
        min_length if min_length is not None
        else config.FILLER_MIN_WORD_LENGTH
    )

    min_words = getattr(config, "MIN_WORDS_FOR_CLEANING", 50)

    # ------------------------------------------------------------------
    # Tokenise for counting
    # Alphabetic tokens only — avoids numbers, underscores, identifiers
    # ------------------------------------------------------------------

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    total_words = len(words)

    if total_words < min_words:
        # File too short — cleaning would damage semantics
        return text

    # ------------------------------------------------------------------
    # Identify filler words
    # ------------------------------------------------------------------

    short_word_counts = Counter(
        word for word in words if len(word) < min_length
    )

    filler_words = {
        word
        for word, count in short_word_counts.items()
        if count / total_words > threshold
    }

    if not filler_words:
        return text

    # ------------------------------------------------------------------
    # Remove fillers from original text (single regex pass)
    # ------------------------------------------------------------------

    pattern = r"\b(?:" + "|".join(re.escape(w) for w in filler_words) + r")\b"

    cleaned_text = re.sub(
        pattern,
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse all whitespace (spaces, tabs, newlines) left by removals
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    cleaned_word_count = len(re.findall(r"\b[a-zA-Z]+\b", cleaned_text.lower()))
    removed_count = total_words - cleaned_word_count

    print(
        f"[Cleaner] Removed {removed_count} filler tokens "
        f"({len(filler_words)} unique: {sorted(filler_words)})"
    )

    return cleaned_text