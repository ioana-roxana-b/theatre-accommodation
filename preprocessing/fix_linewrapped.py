"""
Fix plays where PDNC captured only the first physical line of multi-line speeches.

These plays use an inline format (Name. dialogue) where long speeches wrap across
multiple lines. PDNC captures the first line as the full quote, truncating the rest.

Plays affected:
  - Becque_LesCorbeaux  (FR): "Name[, stage dir]. – dialogue"
  - Dumas_Camille_EN    (EN): "Name. dialogue"

Run:
    python preprocessing/fix_linewrapped.py <play_dir>
"""

import re
import sys
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Per-play configuration: speaker name variants and text format.
# ---------------------------------------------------------------------------
PLAY_CONFIGS = {
    "Becque_LesCorbeaux": {
        "format": "fr_dash",
        # play text: "Name[, stage dir]. – dialogue"
        # Map from form-in-text to canonical name (as in quotation_info)
        "speaker_norms": {
            "Madame Vigneron": "Madame Vigneron",
            "Madame de Saint-Genis": "Madame de Saint-Genis",
            "Blanche etJudith": "Blanche etJudith",
            "Les Enfants": "Les Enfants",
            "Le Medecin": "Le Medecin",
            "Blanche": "Blanche",
            "Marie": "Marie",
            "Judith": "Judith",
            "Vigneron": "Vigneron",
            "Rosalie": "Rosalie",
            "Teissier": "Teissier",
            "Bourdon": "Bourdon",
            "Merckens": "Merckens",
            "Lefort": "Lefort",
            "Dupuis": "Dupuis",
            "Auguste": "Auguste",
            "Gaston": "Gaston",
        },
    },
    "Dumas_Camille_EN": {
        "format": "en_period",
        # play text: "Name. [optional [stage dir]] dialogue"
        "speaker_norms": {
            # canonical forms
            "Armand": "Armand",
            "Camille": "Camille",
            "Duval": "Duval",
            "Gaston": "Gaston",
            "Gustave": "Gustave",
            "Messenger": "Messenger",
            "Nanine": "Nanine",
            "Nichette": "Nichette",
            "Olimpe": "Olimpe",
            "Prudence": "Prudence",
            "Varville": "Varville",
            # OCR variants found as speaker labels in play_text.txt
            "Armani": "Armand",
            "Armancl": "Armand",
            "CamiUe": "Camille",
            "Camtlle": "Camille",
            "CAMILLE": "Camille",
            "G-aston": "Gaston",
            "Gastoyi": "Gaston",
            "NicJiette": "Nichette",
        },
    },
}


def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)       # remove [stage dirs]
    text = re.sub(r"\([^)]*\)", "", text)      # remove (stage dirs)
    text = re.sub(r"[*^]+", "", text)          # remove OCR artifacts
    text = re.sub(r"\s+", " ", text).strip()
    return text


def identify_speaker_fr(stripped: str, sorted_names: list, speaker_norms: dict):
    """
    French dash format: "Name[, stage dir]. – dialogue"
    Returns (canonical_name, dialogue_text) or None.
    """
    for name in sorted_names:
        if not stripped.lower().startswith(name.lower()):
            continue
        rest = stripped[len(name):]
        # Look for any em-dash / en-dash / hyphen after a period
        m = re.search(r"\.\s*[-–—]+\s*(.*)", rest)
        if m:
            return (speaker_norms[name], m.group(1))
    return None


def identify_speaker_en(stripped: str, sorted_names: list, speaker_norms: dict):
    """
    English period format: "Name. [optional [stage dir]] dialogue"
    Returns (canonical_name, dialogue_text) or None.
    """
    for name in sorted_names:
        if not stripped.lower().startswith(name.lower()):
            continue
        rest = stripped[len(name):]
        # Require period + whitespace or period + bracket, then non-empty text
        m = re.match(r"^\.\s*(?:\[.*?\])?\s*(.+)", rest)
        if m:
            return (speaker_norms[name], m.group(1))
    return None


def parse_play(txt_path: Path, speaker_norms: dict, fmt: str) -> list[tuple[str, str]]:
    """Parse play text returning (canonical_speaker, full_text) for every speech."""
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    sorted_names = sorted(speaker_norms.keys(), key=len, reverse=True)
    identify_fn = identify_speaker_fr if fmt == "fr_dash" else identify_speaker_en

    pairs: list[tuple[str, str]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    def flush():
        if current_speaker and current_lines:
            text = clean_text(" ".join(current_lines))
            if text:
                pairs.append((current_speaker, text))
        current_lines.clear()

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush()
            current_speaker = None
            continue

        result = identify_fn(stripped, sorted_names, speaker_norms)
        if result:
            flush()
            current_speaker = result[0]
            text = clean_text(result[1])
            if text:
                current_lines.append(text)
        elif current_speaker:
            text = clean_text(stripped)
            if text:
                current_lines.append(text)

    flush()
    return pairs


def _normt(t: str) -> str:
    """Fully normalize text for matching (lowercase, collapse whitespace, strip OCR artifacts)."""
    t = re.sub(r"[*^]", "", t)           # remove common OCR artifacts
    t = re.sub(r"\([^)]*\)", "", t)      # remove parenthetical stage dirs
    t = re.sub(r"\[[^\]]*\]", "", t)     # remove bracket stage dirs
    return re.sub(r"\s+", " ", t.lower()).strip()


def build_quotation_info(
    play_pairs: list[tuple[str, str]],
    original_qi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rebuild quotation_info in play order with full (untruncated) speech texts.
    Preserves PDNC addressees where an original row can be matched.

    Matching strategy (in order):
      1. Exact normalized match
      2. New text starts with original text (orig is prefix of new — truncation case)
      3. Original text starts with new text (new is shorter — stage-dir stripping)
    Minimum 15-char match to avoid false positives on very short phrases.
    """
    # Build list of (normalized_text, row_dict) from original, sorted longest-first
    orig_entries: list[tuple[str, dict]] = []
    seen_keys: set[str] = set()
    for _, row in original_qi.iterrows():
        nt = _normt(str(row.get("quoteText", "")))
        if nt and nt not in seen_keys:
            seen_keys.add(nt)
            orig_entries.append((nt, row.to_dict()))
    # Sort longest-first so more-specific matches win
    orig_entries.sort(key=lambda x: -len(x[0]))

    MIN_MATCH = 15  # minimum chars for a prefix match

    def find_orig(new_text: str) -> dict | None:
        nt_new = _normt(new_text)
        # Exact match first
        for nt_orig, row in orig_entries:
            if nt_new == nt_orig:
                return row
        # Prefix match: original is a prefix of new (truncation case)
        for nt_orig, row in orig_entries:
            if len(nt_orig) >= MIN_MATCH and nt_new.startswith(nt_orig):
                return row
        # Prefix match: new is a prefix of original (stage-dir stripping case)
        for nt_orig, row in orig_entries:
            if len(nt_new) >= MIN_MATCH and nt_orig.startswith(nt_new):
                return row
        return None

    rows = []
    prev_speaker: str | None = None
    q_counter = 10000

    for speaker, text in play_pairs:
        match = find_orig(text)
        if match:
            orig = match.copy()
            orig["quoteText"] = text   # replace truncated text with full text
            orig["speaker"] = speaker  # use canonical name
            rows.append(orig)
        else:
            # No match — infer addressee from previous speaker
            addressee = (
                prev_speaker
                if prev_speaker and prev_speaker != speaker
                else None
            )
            rows.append({
                "quoteID": f"Q{q_counter}",
                "quoteText": text,
                "subQuotationList": "[]",
                "quoteByteSpans": "[]",
                "speaker": speaker,
                "addressees": f'["{addressee}"]' if addressee else '[]',
                "quoteType": "Explicit",
                "referringExpression": "",
            })
            q_counter += 1
        prev_speaker = speaker

    return pd.DataFrame(rows)


def fix_play(play_dir: Path) -> None:
    play_name = play_dir.name
    if play_name not in PLAY_CONFIGS:
        print(f"No config for '{play_name}'. Nothing to do.")
        return

    config = PLAY_CONFIGS[play_name]
    speaker_norms = config["speaker_norms"]
    fmt = config["format"]

    qi_path = play_dir / "quotation_info.csv"
    txt_path = play_dir / "play_text.txt"

    for p in [qi_path, txt_path]:
        if not p.exists():
            print(f"Missing: {p}")
            return

    # Back up
    bak = qi_path.with_suffix(".csv.bak")
    if bak.exists():
        shutil.copy2(bak, qi_path)
        print(f"Restored from backup: {bak.name}")
    else:
        shutil.copy2(qi_path, bak)
        print(f"Backed up quotation_info.csv")

    # Parse play text
    pairs = parse_play(txt_path, speaker_norms, fmt)
    print(f"Parsed {len(pairs)} speeches from play text")
    print("  Distribution:", dict(Counter(sp for sp, _ in pairs)))

    # Load original
    original_qi = pd.read_csv(qi_path)
    print(f"Original quotation_info: {len(original_qi)} rows")
    avg_words = original_qi["quoteText"].astype(str).str.split().str.len().mean()
    print(f"  Original avg words/quote: {avg_words:.1f}")

    # Rebuild
    new_qi = build_quotation_info(pairs, original_qi)
    print(f"New quotation_info: {len(new_qi)} rows")
    avg_words_new = new_qi["quoteText"].astype(str).str.split().str.len().mean()
    print(f"  New avg words/quote: {avg_words_new:.1f}")
    print("  Distribution:", new_qi["speaker"].value_counts().to_dict())

    new_qi.to_csv(qi_path, index=False)
    print(f"Wrote {qi_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocessing/fix_linewrapped.py <play_dir>")
        sys.exit(1)
    fix_play(Path(sys.argv[1]))
