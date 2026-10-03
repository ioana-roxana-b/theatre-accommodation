"""
Fix The Maids EN: fully reconstruct quotation_info.csv from play_text.txt
to correct OCR-corrupted SOLANGE speaker labels.

Instead of appending rows, we parse the play text in order and build a new
quotation_info in the correct chronological sequence. Addressees are inferred
from the previous speaker (heuristic).

Run:
    python preprocessing/fix_maids_en.py data/plays_dialogue/Genet_TheMaids_EN
"""

import re
import sys
import shutil
from pathlib import Path

import pandas as pd


# Patterns to identify each speaker in the play text
SPEAKER_MAP = [
    ("SOLANGE", re.compile(
        r"^(SoLANGE|So ?LANGE|Sou\.NGE|SOLANGE)(?:\s*\[.*?\])?\s*:?\s*$",
        re.IGNORECASE,
    )),
    ("CLAIRE", re.compile(
        r"^(CLAIRE)(?:\s*\[.*?\])?\s*:?\s*$",
        re.IGNORECASE,
    )),
    ("MADAME", re.compile(
        r"^(MADAME)(?:\s*\[.*?\])?\s*:?\s*$",
        re.IGNORECASE,
    )),
]

# Inline pattern: SPEAKER[...]: dialogue on same line
INLINE_PAT = re.compile(
    r"^(SoLANGE|So ?LANGE|Sou\.NGE|SOLANGE|CLAIRE|MADAME)"
    r"(?:\s*\[.*?\])?\s*:\s*(.+)",
    re.IGNORECASE,
)

STAGE_PAT = re.compile(r"^\[.*?\]$")


def normalize_speaker(raw: str) -> str:
    raw_up = raw.strip().upper().replace(" ", "")
    if re.match(r"SO[A-Z\.]*NGE|SOU\.NGE", raw_up):
        return "SOLANGE"
    if raw_up.startswith("CLAIRE"):
        return "CLAIRE"
    if raw_up.startswith("MADAME"):
        return "MADAME"
    return raw.strip().upper()


def is_speaker_line(stripped: str) -> str | None:
    """Return normalized speaker name if the line is a speaker label, else None."""
    # Inline format won't be a speaker-only line
    m = INLINE_PAT.match(stripped)
    if m and m.group(2).strip():
        return None  # inline line, not a pure speaker label
    for sp_name, pat in SPEAKER_MAP:
        if pat.match(stripped):
            return sp_name
    return None


def parse_play_ordered(txt_path: Path) -> list[tuple[str, str]]:
    """
    Parse play text in order, returning (speaker, full_dialogue_text) for every speech.
    Collects all lines of each speech (not just the first).
    Handles both inline and block formats.
    """
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    pairs = []
    current_speaker = None
    current_lines: list[str] = []

    def flush():
        if current_speaker and current_lines:
            text = " ".join(current_lines).strip()
            if text:
                pairs.append((current_speaker, text))
        current_lines.clear()

    for line in lines:
        stripped = line.strip()

        # Inline: SPEAKER[...]: text (speaker + dialogue on same line)
        m = INLINE_PAT.match(stripped)
        if m and m.group(2).strip():
            flush()
            current_speaker = None
            sp = normalize_speaker(m.group(1))
            text = re.sub(r"\[.*?\]", "", m.group(2)).strip()
            if text:
                pairs.append((sp, text))
            continue

        # Speaker label only
        sp = is_speaker_line(stripped)
        if sp:
            flush()
            current_speaker = sp
            continue

        # Stage direction only — skip (don't add to dialogue)
        if STAGE_PAT.match(stripped):
            continue

        # Blank line — ends the current speech
        if stripped == "":
            flush()
            current_speaker = None
            continue

        # Regular dialogue line
        if current_speaker:
            text_part = re.sub(r"\[.*?\]", "", stripped).strip()
            if text_part:
                current_lines.append(text_part)

    flush()
    return pairs


def build_quotation_info(
    play_pairs: list[tuple[str, str]],
    original_qi: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a new quotation_info.csv in play-text order.
    For lines already in the original, preserve their PDNC addressees.
    For new lines, infer addressee = previous speaker.
    """

    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", t.lower()).strip()[:80]

    # Build lookup: normalized text -> original row (first match wins)
    original_lookup: dict[str, dict] = {}
    for _, row in original_qi.iterrows():
        key = norm(str(row.get("quoteText", "")))
        if key not in original_lookup:
            original_lookup[key] = row.to_dict()

    rows = []
    prev_speaker = None
    q_counter = 10000

    for speaker, text in play_pairs:
        key = norm(text)
        if key in original_lookup:
            rows.append(original_lookup[key])
        else:
            addressee = prev_speaker if prev_speaker and prev_speaker != speaker else (
                "CLAIRE" if speaker != "CLAIRE" else "SOLANGE"
            )
            rows.append({
                "quoteID": f"Q{q_counter}",
                "quoteText": text,
                "subQuotationList": "[]",
                "quoteByteSpans": "[]",
                "speaker": speaker,
                "addressees": f'["{addressee}"]',
                "quoteType": "Explicit",
                "referringExpression": "",
            })
            q_counter += 1
        prev_speaker = speaker

    return pd.DataFrame(rows)


def fix_maids_en(play_dir: Path) -> None:
    qi_path = play_dir / "quotation_info.csv"
    ci_path = play_dir / "character_info.csv"
    txt_path = play_dir / "play_text.txt"

    for p in [qi_path, ci_path, txt_path]:
        if not p.exists():
            print(f"Missing: {p}"); return

    # Always restore from backup if it exists (so we work from the clean original)
    for stem in ["quotation_info.csv", "character_info.csv"]:
        bak = play_dir / (stem + ".bak")
        orig = play_dir / stem
        if bak.exists():
            shutil.copy2(bak, orig)
        else:
            shutil.copy2(orig, bak)
            print(f"Backed up {stem}")

    # Parse play text
    play_pairs = parse_play_ordered(txt_path)
    print(f"Parsed {len(play_pairs)} dialogue lines from play text")
    from collections import Counter
    print("  Distribution:", dict(Counter(sp for sp, _ in play_pairs)))

    # Load original quotation_info
    original_qi = pd.read_csv(qi_path)
    print(f"Original quotation_info: {len(original_qi)} rows")

    # Build new quotation_info in play order
    new_qi = build_quotation_info(play_pairs, original_qi)
    print(f"New quotation_info: {len(new_qi)} rows")
    print("  Distribution:", new_qi["speaker"].value_counts().to_dict())

    new_qi.to_csv(qi_path, index=False)

    # Fix character_info: expand SOLANGE aliases and set category to major
    ci = pd.read_csv(ci_path)
    mask = ci["Main Name"] == "SOLANGE"
    if mask.any():
        ci.loc[mask, "Aliases"] = str({"SOLANGE", "SO LANGE", "SoLANGE", "Sou.NGE"})
        ci.loc[mask, "Category"] = "major"
    ci.to_csv(ci_path, index=False)
    print("Updated character_info.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocessing/fix_maids_en.py <play_dir>")
        sys.exit(1)
    fix_maids_en(Path(sys.argv[1]))
