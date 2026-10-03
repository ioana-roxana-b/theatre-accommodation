"""
Fix remaining realist play annotation issues:

1. Brieux_LaRobeRouge_EN: 'The Recorder' should be 'Recorder' (same character, PDNC split).
2. Brieux_WomanOnHerOwn:  'A Man'/'A Woman' and 'Another Man'/'Another Woman' were
   merged by PDNC but are distinct characters. Also 'Maud'/'Maid' conflation.

Run:
    python preprocessing/fix_realists.py <play_dir>
"""

import re
import sys
import shutil
from pathlib import Path

import pandas as pd


def fix_robe_rouge_en(play_dir: Path) -> None:
    """Merge 'The Recorder' -> 'Recorder' in quotation_info and character_info."""
    qi_path = play_dir / "quotation_info.csv"
    ci_path = play_dir / "character_info.csv"

    for p in [qi_path, ci_path]:
        bak = p.with_suffix(p.suffix + ".bak")
        if bak.exists():
            shutil.copy2(bak, p)
        else:
            shutil.copy2(p, bak)
            print(f"Backed up {p.name}")

    # Fix quotation_info: rename The Recorder -> Recorder
    qi = pd.read_csv(qi_path)
    mask = qi["speaker"] == "The Recorder"
    print(f"  Renaming {mask.sum()} 'The Recorder' turns -> 'Recorder'")
    qi.loc[mask, "speaker"] = "Recorder"
    qi.to_csv(qi_path, index=False)
    print(f"  Speaker counts: {qi['speaker'].value_counts().to_dict()}")

    # Fix character_info: remove The Recorder row, add alias to Recorder
    ci = pd.read_csv(ci_path)
    the_recorder_rows = ci[ci["Main Name"] == "The Recorder"]
    recorder_rows = ci[ci["Main Name"] == "Recorder"]

    if not the_recorder_rows.empty and not recorder_rows.empty:
        ci = ci[ci["Main Name"] != "The Recorder"].reset_index(drop=True)
        # Add 'The Recorder' as alias
        idx = ci[ci["Main Name"] == "Recorder"].index[0]
        aliases_raw = str(ci.at[idx, "Aliases"])
        aliases = set(re.findall(r"'([^']+)'", aliases_raw))
        aliases.add("The Recorder")
        ci.at[idx, "Aliases"] = str(aliases)
        ci.to_csv(ci_path, index=False)
        print(f"  Merged 'The Recorder' into 'Recorder' aliases. character_info now {len(ci)} rows.")
    else:
        print(f"  Warning: could not find both Recorder rows in character_info.")


def fix_woman_on_her_own(play_dir: Path) -> None:
    """
    Split A Man / A Woman and Another Man / Another Woman in character_info.
    These are distinct characters that PDNC merged as aliases.
    Also split Maud / Maid.

    Since these characters have very few turns (A Man: 9, Another Man: 6, Maud: 2),
    we only fix character_info (remove the wrong alias). We cannot split the
    quotation_info speaker labels without knowing which turns belong to which
    character from the play text — and with such low counts these characters
    won't appear in any min_support=3 dyad.
    """
    ci_path = play_dir / "character_info.csv"
    bak = ci_path.with_suffix(ci_path.suffix + ".bak")
    if bak.exists():
        shutil.copy2(bak, ci_path)
    else:
        shutil.copy2(ci_path, bak)
        print(f"Backed up character_info.csv")

    ci = pd.read_csv(ci_path)

    # Remove cross-gender aliases
    fixes = {
        "A Man": "A Woman",
        "Another Man": "Another Woman",
        "Maud": "Maid",
    }

    for main_name, wrong_alias in fixes.items():
        mask = ci["Main Name"] == main_name
        if not mask.any():
            continue
        idx = ci[mask].index[0]
        aliases_raw = str(ci.at[idx, "Aliases"])
        aliases = set(re.findall(r"'([^']+)'", aliases_raw))
        if wrong_alias in aliases:
            aliases.discard(wrong_alias)
            ci.at[idx, "Aliases"] = str(aliases)
            print(f"  Removed '{wrong_alias}' from aliases of '{main_name}'")

    ci.to_csv(ci_path, index=False)
    print(f"  Updated character_info.csv ({len(ci)} rows)")


PLAY_FIXES = {
    "Brieux_LaRobeRouge_EN": fix_robe_rouge_en,
    "Brieux_WomanOnHerOwn": fix_woman_on_her_own,
}


def main(play_dir: Path) -> None:
    play_name = play_dir.name
    if play_name not in PLAY_FIXES:
        print(f"No fix defined for '{play_name}'. Options: {list(PLAY_FIXES)}")
        return
    print(f"Fixing {play_name}...")
    PLAY_FIXES[play_name](play_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocessing/fix_realists.py <play_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
