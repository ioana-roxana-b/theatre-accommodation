"""
Stopword-based feature extraction.
Treats each stopword as its own convergence family, using the stopwords-iso list
for the target language.  The output CSV format is identical to features.py so the
entire downstream pipeline (convergence, visualisation, graph, stats) is reused
without modification.
"""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd
import stopwordsiso as swiso

_SW_CACHE: dict = {}


def _get_stopwords(lang: str) -> frozenset:
    if lang not in _SW_CACHE:
        _SW_CACHE[lang] = frozenset(swiso.stopwords(lang))
    return _SW_CACHE[lang]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿÀ-ɏ]+", text.lower())


def count_stopword_features(text: str, stopwords: frozenset) -> Dict[str, int]:
    tokens = _tokenize(text if isinstance(text, str) else "")
    counts = Counter(t for t in tokens if t in stopwords)
    return counts, len(tokens)


def process_pairs(pairs_path: Path, out_path: Path, lang: str = "en") -> pd.DataFrame:
    """
    Build per-pair stopword features. Each stopword becomes its own family column
    (*_i_count, *_j_count, *_i_present, *_j_present), compatible with the spaCy
    convergence scorer.
    """
    sw = _get_stopwords(lang)
    sw_sorted = sorted(sw)

    pairs = pd.read_csv(pairs_path)
    need = ["A", "B", "i_text", "j_text"]
    miss = [c for c in need if c not in pairs.columns]
    if miss:
        raise ValueError(f"Pairs file missing columns: {miss}")

    rows = []
    for _, r in pairs.iterrows():
        ci, i_ntok = count_stopword_features(r["i_text"], sw)
        cj, j_ntok = count_stopword_features(r["j_text"], sw)

        row: dict = {
            "A": r["A"],
            "B": r["B"],
            "i_text": r["i_text"],
            "j_text": r["j_text"],
        }
        for word in sw_sorted:
            row[f"{word}_i_count"] = ci.get(word, 0)
            row[f"{word}_j_count"] = cj.get(word, 0)
            row[f"{word}_i_present"] = int(ci.get(word, 0) > 0)
            row[f"{word}_j_present"] = int(cj.get(word, 0) > 0)

        row["i_tokens"] = i_ntok
        row["j_tokens"] = j_ntok

        for extra in ["i_merge_id", "j_merge_id", "i_addressees", "distance", "i_quoteID", "j_quoteID"]:
            if extra in pairs.columns:
                row[extra] = r[extra]

        for c in pairs.columns:
            if c.startswith("A_") or c.startswith("B_"):
                row[c] = r.get(c)

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")
    return out
