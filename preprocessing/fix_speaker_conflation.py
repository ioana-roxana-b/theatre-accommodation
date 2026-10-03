"""
Fix PDNC speaker conflation in plays where male/female characters with the same
surname were incorrectly merged into one entity.

Usage:
    python preprocessing/fix_speaker_conflation.py <play_dialogue_dir>

Example:
    python preprocessing/fix_speaker_conflation.py data/plays_dialogue/Ionesco_BaldSoprano_EN

The script:
  1. Reads the play_text.txt and builds a (cleaned_text_prefix -> correct_speaker) map.
  2. Reads quotation_info.csv and corrects speaker labels using the map.
  3. Reads character_info.csv and un-merges aliased characters.
  4. Writes corrected files back in-place (original backed up with .bak extension).
"""

import re
import sys
import shutil
import unicodedata
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Known conflations: map alias -> canonical for each play.
# Key = play directory name, value = dict{alias: correct_name, ...}
# We also record the gender/category for newly split-out characters.
# ---------------------------------------------------------------------------
CONFLATIONS = {
    "Ionesco_BaldSoprano_EN": {
        "split": [
            # (merged_canonical, alias_that_is_actually_different, gender, category)
            ("MRS. SMITH",  "MR. SMITH",   "M", "intermediate"),
            ("MR. MARTIN",  "MRS. MARTIN", "F", "minor"),
        ],
        "remove": ["MR. AND MRS. SMITH"],
    },
    "Genet_Balcony_EN": {
        "split": [
            # THE WOMAN absorbed THE MAN (different scenes, different characters)
            ("THE WOMAN", "THE MAN", "M", "minor"),
        ],
        "remove": [],
    },
}

# Known characters that appear as speakers in the play text for each play.
KNOWN_SPEAKERS = {
    "Ionesco_BaldSoprano_EN": {
        "MR. SMITH", "MRS. SMITH", "MR. MARTIN", "MRS. MARTIN",
        "MARY", "FIRE CHIEF", "THE FIRE CHIEF", "ALL TOGETHER",
    },
    "Genet_Balcony_EN": {
        "THE WOMAN", "THE MAN", "IRMA", "CARMEN", "THE CHIEF OF POLICE",
        "ROGER", "THE ENVOY", "THE BISHOP", "THE GENERAL", "THE JUDGE",
        "THE QUEEN", "ARTHUR", "THE THIEF", "CHANTAL", "THE EXECUTIONER",
        "THE SLAVE", "ONE OF THE REBELS",
    },
}


def normalize(text: str) -> str:
    """Lowercase, collapse whitespace, remove punctuation for fuzzy matching."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_play_text(play_txt: Path, known_speakers: set) -> list[tuple[str, str]]:
    """
    Parse play_text.txt and return list of (speaker, dialogue_text) for every
    spoken line. Handles two formats:
      Format 1 (inline): "SPEAKER: dialogue text"
      Format 2 (block):  speaker name on own line, dialogue on subsequent line(s)
    Stage-direction-only lines are skipped.
    """
    escaped = sorted(known_speakers, key=len, reverse=True)
    speaker_pat = re.compile(
        r"^(" + "|".join(re.escape(s) for s in escaped) + r")\s*(.*)",
        re.IGNORECASE,
    )
    stage_pat = re.compile(r"^\[.*?\]\s*$")

    raw_lines = play_txt.read_text(encoding="utf-8").splitlines()
    pairs = []
    current_speaker = None
    collecting = False

    for raw_line in raw_lines:
        line = raw_line.strip()
        m = speaker_pat.match(line)

        if m:
            speaker = m.group(1).upper()
            rest = m.group(2).strip()
            # Remove inline stage directions
            rest = re.sub(r"\[.*?\]", "", rest).strip()

            if rest.startswith(":"):
                # Format 1: inline dialogue
                dialogue = rest[1:].strip()
                if dialogue:
                    pairs.append((speaker, dialogue))
                current_speaker = None
                collecting = False
            elif rest == "":
                # Format 2: speaker name only, dialogue follows
                current_speaker = speaker
                collecting = True
            else:
                # Speaker name followed by non-colon text — treat as dialogue
                if rest:
                    pairs.append((speaker, rest))
                current_speaker = None
                collecting = False
        elif collecting and current_speaker:
            if line == "" or stage_pat.match(line):
                # blank line or stage direction — keep waiting for dialogue
                pass
            else:
                # This is the dialogue line
                dialogue = re.sub(r"\[.*?\]", "", line).strip()
                if dialogue:
                    pairs.append((current_speaker, dialogue))
                collecting = False
                current_speaker = None

    return pairs


def build_text_to_speaker(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """
    Build mapping from normalized text prefix (first 60 chars) -> speaker.
    Only includes entries where the prefix is unique (no ambiguous short lines).
    """
    prefix_len = 60
    mapping: dict[str, list[str]] = {}
    for speaker, text in pairs:
        key = normalize(text)[:prefix_len]
        mapping.setdefault(key, []).append(speaker)

    # Keep only unambiguous entries
    return {k: v[0] for k, v in mapping.items() if len(set(v)) == 1}


def correct_quotation_info(
    qi_path: Path,
    text_to_speaker: dict[str, str],
    conflations: list[tuple[str, str, str, str]],
) -> pd.DataFrame:
    """
    Read quotation_info.csv and return a corrected copy with fixed speaker labels.
    """
    # The set of wrong-speaker names we might want to correct
    wrong_speakers = {merged for merged, _, _, _ in conflations}

    df = pd.read_csv(qi_path)
    corrections = 0

    for idx, row in df.iterrows():
        sp = str(row["speaker"]).strip()
        if sp not in wrong_speakers:
            continue
        raw_text = str(row.get("quoteText", "")).strip()
        key = normalize(raw_text)[:60]
        correct = text_to_speaker.get(key)
        if correct and correct != sp:
            df.at[idx, "speaker"] = correct
            corrections += 1

    print(f"  Corrected {corrections} speaker labels in quotation_info.csv")
    return df


def correct_character_info(
    ci_path: Path,
    conflations: list[tuple[str, str, str, str]],
    remove_names: list[str],
) -> pd.DataFrame:
    """
    Read character_info.csv and return a corrected copy with split-out characters.
    """
    df = pd.read_csv(ci_path)

    new_rows = []
    rows_to_drop = []

    for i, row in df.iterrows():
        main_name = str(row.get("Main Name", "")).strip()

        # Remove spurious merged entries
        if main_name in remove_names:
            rows_to_drop.append(i)
            continue

        # Check if this row has an alias that should be a separate character
        for merged, split_name, gender, category in conflations:
            if main_name != merged:
                continue
            # Clean up the aliases for the merged character
            aliases_raw = str(row.get("Aliases", ""))
            # Parse the set-like string
            aliases = set(re.findall(r"'([^']+)'", aliases_raw))
            # Remove the split-out alias from this row's aliases
            aliases.discard(split_name)
            df.at[i, "Aliases"] = str(aliases)

            # Add a new row for the split-out character
            new_row = row.copy()
            new_row["Main Name"] = split_name
            new_row["Aliases"] = str({split_name})
            new_row["Gender"] = gender
            new_row["Category"] = category
            # Assign a new Character ID (max + offset)
            max_id = df["Character ID"].max()
            new_row["Character ID"] = max_id + 1 + len(new_rows)
            new_rows.append(new_row)

    df = df.drop(index=rows_to_drop).reset_index(drop=True)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    print(f"  Added {len(new_rows)} new character rows, removed {len(rows_to_drop)} merged rows")
    return df


def fix_play(play_dir: Path) -> None:
    play_name = play_dir.name
    if play_name not in CONFLATIONS:
        print(f"No conflation rules defined for '{play_name}'. Nothing to do.")
        return

    config = CONFLATIONS[play_name]
    conflations = config["split"]
    remove_names = config.get("remove", [])
    known_speakers = KNOWN_SPEAKERS.get(play_name, set())

    play_txt = play_dir / "play_text.txt"
    qi_path = play_dir / "quotation_info.csv"
    ci_path = play_dir / "character_info.csv"

    for p in [play_txt, qi_path, ci_path]:
        if not p.exists():
            print(f"Missing file: {p}")
            return

    print(f"Fixing {play_name}...")

    # 1. Parse play text
    pairs = parse_play_text(play_txt, known_speakers)
    text_to_speaker = build_text_to_speaker(pairs)
    print(f"  Parsed {len(pairs)} dialogue lines, {len(text_to_speaker)} unique prefixes")

    # 2. Back up originals
    for p in [qi_path, ci_path]:
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(p, bak)
            print(f"  Backed up {p.name} -> {bak.name}")

    # 3. Correct quotation_info.csv
    qi_fixed = correct_quotation_info(qi_path, text_to_speaker, conflations)
    qi_fixed.to_csv(qi_path, index=False)
    print(f"  Wrote corrected quotation_info.csv ({len(qi_fixed)} rows)")
    print("  New speaker counts:")
    for sp, cnt in qi_fixed["speaker"].value_counts().items():
        print(f"    {sp}: {cnt}")

    # 4. Correct character_info.csv
    ci_fixed = correct_character_info(ci_path, conflations, remove_names)
    ci_fixed.to_csv(ci_path, index=False)
    print(f"  Wrote corrected character_info.csv ({len(ci_fixed)} rows)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocessing/fix_speaker_conflation.py <play_dialogue_dir>")
        sys.exit(1)
    fix_play(Path(sys.argv[1]))
