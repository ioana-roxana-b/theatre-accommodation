"""
Word-list-based feature extraction for the 9 LIWC-aligned families.
Tokenises text by splitting on whitespace and non-alpha characters,
handling French elision apostrophes so that "l'homme" yields ["l", "homme"].
No spaCy POS tagging — membership is determined purely by the explicit lists
in function_word_lists.py.
"""
from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from preprocessing.function_word_lists import LISTS

FAMILIES = list(next(iter(LISTS.values())).keys())  # same 9 keys for all langs


def _tokenize(text: str) -> list[str]:
    """
    Lowercase and split on whitespace + non-alpha chars.
    Apostrophes are treated as separators so elision fragments
    (l, d, j, n, qu, m, t, s, c) become their own tokens.
    """
    return re.findall(r"[a-zA-ZÀ-ÿÀ-ɏàâçéèêëîïôùûüÿæœ]+", text.lower())


def count_families(text: str, lang: str = "en") -> Dict[str, int]:
    families = LISTS.get(lang, LISTS["en"])
    counts: Dict[str, int] = {k: 0 for k in families}
    tokens = _tokenize(text if isinstance(text, str) else "")
    for tok in tokens:
        for fam, word_set in families.items():
            if tok in word_set:
                counts[fam] += 1
    counts["n_tokens"] = len(tokens)
    return counts


def process_pairs(pairs_path: Path, out_path: Path, lang: str = "en") -> pd.DataFrame:
    """Apply word-list family detection to dialogue pairs (same output format as features.py)."""
    pairs = pd.read_csv(pairs_path)
    need = ["A", "B", "i_text", "j_text"]
    miss = [c for c in need if c not in pairs.columns]
    if miss:
        raise ValueError(f"Pairs file missing columns: {miss}")

    families = LISTS.get(lang, LISTS["en"])
    rows = []
    for _, r in pairs.iterrows():
        ci = count_families(r["i_text"], lang)
        cj = count_families(r["j_text"], lang)

        row: dict = {
            "A": r["A"],
            "B": r["B"],
            "i_text": r["i_text"],
            "j_text": r["j_text"],
        }
        for fam in families:
            row[f"{fam}_i_count"] = ci[fam]
            row[f"{fam}_j_count"] = cj[fam]
            row[f"{fam}_i_present"] = int(ci[fam] > 0)
            row[f"{fam}_j_present"] = int(cj[fam] > 0)

        row["i_tokens"] = ci["n_tokens"]
        row["j_tokens"] = cj["n_tokens"]

        # Question: punctuation check on raw text (tokeniser strips ?)
        row["question_i_present"] = int("?" in str(r["i_text"]))
        row["question_j_present"] = int("?" in str(r["j_text"]))

        for extra in ["i_merge_id", "j_merge_id", "i_addressees", "distance", "i_quoteID", "j_quoteID"]:
            if extra in pairs.columns:
                row[extra] = r[extra]

        for c in pairs.columns:
            if c.startswith("A_") or c.startswith("B_"):
                row[c] = r.get(c)

        rows.append(row)

    # Length bucket: short = below-median token count across all turns in this file
    if rows:
        all_lengths = [r["i_tokens"] for r in rows] + [r["j_tokens"] for r in rows]
        median_len = float(np.median(all_lengths))
        for r in rows:
            r["length_short_i_present"] = int(r["i_tokens"] <= median_len)
            r["length_short_j_present"] = int(r["j_tokens"] <= median_len)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")
    return out
