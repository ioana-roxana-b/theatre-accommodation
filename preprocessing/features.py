import spacy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

_NLP_CACHE: dict = {}

_ARTICLES = {
    "en": frozenset({"a", "an", "the"}),
    "fr": frozenset({"le", "la", "les", "un", "une", "des", "du", "au", "aux", "l"}),
}
# French ADV negators (ne...pas, ne...jamais, etc.)
_NEG_ADV_FR = frozenset({"pas", "ne", "jamais", "plus", "guère", "nullement", "point", "ni"})
_MODELS = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
}


def _get_nlp(lang: str):
    if lang not in _NLP_CACHE:
        model_name = _MODELS.get(lang, f"{lang}_core_news_sm")
        loaded = spacy.load(model_name, disable=["ner"])
        try:
            loaded.enable_pipe("senter")
        except Exception:
            pass
        _NLP_CACHE[lang] = loaded
    return _NLP_CACHE[lang]


# Load default English model at import time for backwards compatibility
nlp = _get_nlp("en")


def is_article(t, articles=_ARTICLES["en"]) -> bool:
    return t.pos_ == "DET" and t.lemma_.lower() in articles

def is_aux(t) -> bool:
    return t.pos_ == "AUX"

def is_conj(t) -> bool:
    return t.pos_ in {"CCONJ", "SCONJ"}

def is_hi_adv(t) -> bool:
    return t.pos_ == "ADV"

def is_neg(t, lang="en") -> bool:
    if t.dep_ == "neg":
        return True
    pron_type = t.morph.get("PronType")
    if t.pos_ in {"DET", "PRON"} and "Neg" in pron_type:
        return True
    if lang == "fr":
        if "Neg" in t.morph.get("Polarity"):
            return True
        if t.pos_ == "ADV" and t.lemma_.lower() in _NEG_ADV_FR:
            return True
    return False

def is_prep(t) -> bool:
    return t.pos_ == "ADP"

def is_pers_pron(t) -> bool:
    return t.pos_ == "PRON" and "Prs" in t.morph.get("PronType")

def is_imp_pron(t) -> bool:
    if t.pos_ in {"PRON", "DET"}:
        pt = set(t.morph.get("PronType"))
        if pt & {"Dem", "Ind", "Tot", "Neg", "Rel"}:
            return True
        if t.pos_ == "PRON":
            person = set(t.morph.get("Person"))
            number = set(t.morph.get("Number"))
            if "3" in person and "Sing" in number:
                return True
    return False

def is_quant(t) -> bool:
    if t.pos_ in {"DET", "PRON"}:
        pt = set(t.morph.get("PronType"))
        if pt & {"Tot", "Ind"}:
            return True
    return False

FNS = {
    "article": is_article,
    "auxverb": is_aux,
    "conj": is_conj,
    "adverb": is_hi_adv,
    "ipron": is_imp_pron,
    "negate": is_neg,
    "ppron": is_pers_pron,
    "prep": is_prep,
    "quant": is_quant,
}

# ── Doc-level syntactic features ─────────────────────────────────────────────

SYNTACTIC_FAMILIES = ("question", "passive", "subord")


def _has_question(doc) -> bool:
    return "?" in doc.text


def _has_passive(doc) -> bool:
    for tok in doc:
        if tok.dep_ in {"nsubjpass", "auxpass", "nsubj:pass"}:
            return True
        if "Pass" in tok.morph.get("Voice"):
            return True
    return False


def _has_subordinate(doc) -> bool:
    # Mark dependency marks subordinating conjunctions (e.g. "because", "although")
    for tok in doc:
        if tok.dep_ == "mark" and tok.pos_ == "SCONJ":
            return True
    return False


def count_families(text: str, lang: str = "en") -> Dict[str, int]:
    """Count occurrences of each LIWC-aligned linguistic family in a text using spaCy POS and morphology."""
    nlp_model = _get_nlp(lang)
    articles = _ARTICLES.get(lang, _ARTICLES["en"])
    local_fns = {**FNS,
                 "article": lambda t: is_article(t, articles),
                 "negate": lambda t: is_neg(t, lang)}

    counts = {k: 0 for k in local_fns}
    n_tokens = 0
    doc = nlp_model(text if isinstance(text, str) else "")
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        n_tokens += 1
        for name, fn in local_fns.items():
            if fn(t):
                counts[name] += 1

    counts["question"] = int(_has_question(doc))
    counts["passive"]  = int(_has_passive(doc))
    counts["subord"]   = int(_has_subordinate(doc))
    counts["n_tokens"] = n_tokens
    return counts


def process_pairs(pairs_path: Path, out_path: Path, lang: str = "en") -> pd.DataFrame:
    """Apply family-level feature extraction to dialogue pairs and write results for CAT computation."""
    pairs = pd.read_csv(pairs_path)

    need = ["A", "B", "i_text", "j_text"]
    miss = [c for c in need if c not in pairs.columns]
    if miss:
        raise ValueError(f"Pairs file missing columns: {miss}")

    all_families = list(FNS.keys()) + list(SYNTACTIC_FAMILIES)

    rows = []
    for _, r in pairs.iterrows():
        ci = count_families(r["i_text"], lang)
        cj = count_families(r["j_text"], lang)

        row = {
            "A": r["A"],
            "B": r["B"],
            "i_text": r["i_text"],
            "j_text": r["j_text"],
        }
        for fam in all_families:
            row[f"{fam}_i_count"] = ci[fam]
            row[f"{fam}_j_count"] = cj[fam]
            row[f"{fam}_i_present"] = int(ci[fam] > 0)
            row[f"{fam}_j_present"] = int(cj[fam] > 0)

        row["i_tokens"] = ci["n_tokens"]
        row["j_tokens"] = cj["n_tokens"]

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
