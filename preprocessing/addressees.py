from __future__ import annotations

import pandas as pd


def window_addressees(speakers: list, window: int = 10) -> list[list[str]]:
    """Addressees per utterance: sorted names of the other speakers within +/- window."""
    n = len(speakers)
    out = []
    for i, spk in enumerate(speakers):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        present = {speakers[j] for j in range(lo, hi)
                   if j != i and pd.notna(speakers[j]) and speakers[j] != spk}
        out.append(sorted(present))
    return out


def recompute_addressees(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Return a copy of a quotation_info table with the addressees column recomputed.
    Rows must be in the order the utterances occur in the play."""
    df = df.copy()
    df["addressees"] = [str(a) for a in window_addressees(df["speaker"].tolist(), window)]
    return df
