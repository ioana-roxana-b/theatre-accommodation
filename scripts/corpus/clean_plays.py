"""Clean PLAYS text files, retaining only actual literary content."""
import re
from pathlib import Path

PLAYS_DIR = Path("data/plays_txt")
OUTPUT_DIR = Path("data/plays_clean")
OUTPUT_DIR.mkdir(exist_ok=True)


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_file(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def is_ocr_garbage(line):
    stripped = line.strip()
    if not stripped:
        return False
    alpha = sum(1 for c in stripped if c.isalpha())
    total = len(stripped)
    if "©" in stripped or "¥" in stripped:
        return True
    if total <= 4 and alpha <= 1:
        return True
    real_words = re.findall(r"[a-zA-ZÀ-ÿœŒæÆ]{2,}", stripped)
    if total <= 8:
        if not real_words:
            return True
        if all(len(w) <= 2 for w in real_words) and alpha <= 3:
            return True
    if total > 8 and not real_words:
        return True
    if total > 3 and alpha / total < 0.20:
        return True
    return False


def is_page_header(line, title):
    """True if line is only the running title + page number / OCR noise."""
    stripped = line.strip()
    if title.lower() not in stripped.lower():
        return False
    remainder = re.sub(re.escape(title), "", stripped, flags=re.IGNORECASE).strip()
    if not remainder:
        return True
    real_words = re.findall(r"[a-zA-ZÀ-ÿœŒæÆ]{3,}", remainder)
    if not real_words:
        return True
    if all(len(w) <= 2 for w in real_words) and len(remainder) <= 10:
        return True
    return False


def find_first_line(lines, marker, start=0):
    for i, line in enumerate(lines[start:], start):
        if marker in line:
            return i
    return start


def remove_running_headers(lines, titles):
    result = []
    for line in lines:
        if any(is_page_header(line, t) for t in titles):
            continue
        result.append(line)
    return result


def remove_pattern_lines(lines, patterns):
    """Remove lines matching any of the given regex patterns."""
    result = []
    for line in lines:
        stripped = line.strip()
        if any(re.match(p, stripped, re.IGNORECASE) for p in patterns):
            continue
        result.append(line)
    return result


def normalize_spaces(lines):
    """Collapse runs of spaces to a single space within each line."""
    return [re.sub(r" {2,}", " ", l) for l in lines]


def dedupe_blanks(lines, max_consecutive=2):
    result = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= max_consecutive:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return result


def clean(lines, start_marker=None, end_marker=None,
          header_titles=None, extra_patterns=None,
          remove_garbage=True, remove_separators=True,
          skip_from=None, skip_to=None, norm_spaces=False):
    if start_marker:
        idx = find_first_line(lines, start_marker)
        lines = lines[idx:]
    if end_marker:
        idx = find_first_line(lines, end_marker)
        lines = lines[:idx + 1]
    if skip_from and skip_to:
        sec_start = find_first_line(lines, skip_from)
        sec_end = find_first_line(lines, skip_to, sec_start)
        lines = lines[:sec_start] + lines[sec_end:]
    if norm_spaces:
        lines = normalize_spaces(lines)
    if remove_garbage:
        lines = [l for l in lines if not is_ocr_garbage(l)]
    if header_titles:
        lines = remove_running_headers(lines, header_titles)
    if extra_patterns:
        lines = remove_pattern_lines(lines, extra_patterns)
    if remove_separators:
        lines = [l for l in lines
                 if not re.match(r"^[-—_=*]{3,}\s*$", l.strip())]
    lines = dedupe_blanks(lines)
    return lines


# ---------------------------------------------------------------------------
# File-specific processing
# ---------------------------------------------------------------------------

# 1. Beckett — En attendant Godot  (clean from line 1)
lines = read_file(PLAYS_DIR / "Beckett_En attendant Godot.txt")
lines = clean(lines)
write_file(OUTPUT_DIR / "Beckett_En attendant Godot.txt", lines)
print("Done: Beckett_En attendant Godot.txt")


# 2. Beckett — Endgame  (long preface; play starts at "Bare interior.")
lines = read_file(PLAYS_DIR / "Beckett_Endgame.txt")
lines = clean(lines, start_marker="Bare interior.")
write_file(OUTPUT_DIR / "Beckett_Endgame.txt", lines)
print("Done: Beckett_Endgame.txt")


# 3. Beckett — Fin de partie  (dedication + cast; play body starts line ~57)
lines = read_file(PLAYS_DIR / "Beckett_Fin de partie.txt")
lines = clean(lines, start_marker="Pour Roger Blin")
write_file(OUTPUT_DIR / "Beckett_Fin de partie.txt", lines)
print("Done: Beckett_Fin de partie.txt")


# 4. Beckett — Waiting for Godot  (E-text header + mangled char list)
lines = read_file(PLAYS_DIR / "Beckett_Waiting for Godot.txt")
lines = clean(lines, start_marker="ACT I")
write_file(OUTPUT_DIR / "Beckett_Waiting for Godot.txt", lines)
print("Done: Beckett_Waiting for Godot.txt")


# 5. Becque — La parisienne  (clean from line 1)
lines = read_file(PLAYS_DIR / "Becque_la parisienne.txt")
lines = clean(lines)
write_file(OUTPUT_DIR / "Becque_la parisienne.txt", lines)
print("Done: Becque_la parisienne.txt")


# 6. Becque — The Vultures  (clean from line 1)
lines = read_file(PLAYS_DIR / "Becque_The vultures.txt")
lines = clean(lines)
write_file(OUTPUT_DIR / "Becque_The vultures.txt", lines)
print("Done: Becque_The vultures.txt")


# 7. Becque — Les corbeaux  (remove the edition "Note" about asterisks)
lines = read_file(PLAYS_DIR / "Becque_les corbeaux.txt")
lines = clean(lines, skip_from="Note\n", skip_to="LES CORBEAUX\n",
              header_titles=["LES CORBEAUX", "CORBEAUX"])
write_file(OUTPUT_DIR / "Becque_les corbeaux.txt", lines)
print("Done: Becque_les corbeaux.txt")


# 8. Becque — The Woman of Paris  (embedded running headers with garbled title)
lines = read_file(PLAYS_DIR / "Becque_the woman of paris.txt")
lines = clean(lines,
              header_titles=["THE WOMAN OF PARIS"],
              extra_patterns=[
                  r"^[\w]{0,3}\s*\d+\s+THE WOMAN OF[.\s\-]*PARIS",
              ])
write_file(OUTPUT_DIR / "Becque_the woman of paris.txt", lines)
print("Done: Becque_the woman of paris.txt")


# 9. Brieux — Woman on Her Own + Red Robe  (clean from line 1)
lines = read_file(PLAYS_DIR / "Brieux_Woman on Her Own+Red robe.txt")
lines = clean(lines)
write_file(OUTPUT_DIR / "Brieux_Woman on Her Own+Red robe.txt", lines)
print("Done: Brieux_Woman on Her Own+Red robe.txt")


# 10. Brieux — Woman on Her Own  (clean from line 1)
lines = read_file(PLAYS_DIR / "Brieux_Woman on Her Own.txt")
lines = clean(lines)
write_file(OUTPUT_DIR / "Brieux_Woman on Her Own.txt", lines)
print("Done: Brieux_Woman on Her Own.txt")


# 11. Dumas — Camille (PLAYS version, double-spaced)
lines = read_file(PLAYS_DIR / "Dumas_Camille.txt")
lines = clean(lines, norm_spaces=True)
write_file(OUTPUT_DIR / "Dumas_Camille.txt", lines)
print("Done: Dumas_Camille.txt")


# 12. Dumas — La Dame aux camélias  (double-spaced + embedded page headers)
lines = read_file(PLAYS_DIR / "Dumas_Dame aux camelias.txt")
lines = clean(lines, norm_spaces=True,
              header_titles=["LA DAME AUX CAMÉLIAS", "LA DAME AUX CAMELIAS"])
write_file(OUTPUT_DIR / "Dumas_Dame aux camelias.txt", lines)
print("Done: Dumas_Dame aux camelias.txt")


# 13. Dumas — Le demi-monde  (double-spaced, Internet Archive + editor INTRODUCTION)
# Keep PERSONNAGES section, skip INTRODUCTION, start play at ACTE PREMIER.
lines = read_file(PLAYS_DIR / "Dumas_Le demi monde.txt")
pers_start = find_first_line(lines, "PERSONNAGES")
intro_start = find_first_line(lines, "INTRODUCTION", pers_start)
acte_start = find_first_line(lines, "ACTE    PREMIER")
char_section = lines[pers_start:intro_start]
play_section = lines[acte_start:]
char_section = normalize_spaces([l for l in char_section if not is_ocr_garbage(l)])
play_section = clean(play_section, norm_spaces=True,
                     header_titles=["ACTE PREMIER", "LE DEMI-MONDE", "DEMI-MONDE"])
lines = char_section + ["\n"] + play_section
lines = dedupe_blanks(lines)
write_file(OUTPUT_DIR / "Dumas_Le demi monde.txt", lines)
print("Done: Dumas_Le demi monde.txt")


# 14. Genet — Balcony file (actually contains The Maids + Deathwatch)
# Long Sartre introduction precedes the plays; start at "THE MAIDS".
lines = read_file(PLAYS_DIR / "Genet_Balcony.txt")
lines = clean(lines, start_marker="THE MAIDS\n")
write_file(OUTPUT_DIR / "Genet_Balcony.txt", lines)
print("Done: Genet_Balcony.txt")


# 15. Genet — The Maids  (clean from line 1; standalone page numbers removed by garbage filter)
lines = read_file(PLAYS_DIR / "Genet_The maids.txt")
lines = clean(lines)
write_file(OUTPUT_DIR / "Genet_The maids.txt", lines)
print("Done: Genet_The maids.txt")


# 16. Ionesco — The Bald Soprano  (publisher preamble; content from "Anti-play"; ToC at end)
lines = read_file(PLAYS_DIR / "Ionesco_Bald Soprano.txt")
lines = clean(lines, start_marker="Anti-play", end_marker="Table of Contents")
lines = [l for l in lines if "Table of Contents" not in l]
lines = dedupe_blanks(lines)
write_file(OUTPUT_DIR / "Ionesco_Bald Soprano.txt", lines)
print("Done: Ionesco_Bald Soprano.txt")


# 17. Ionesco — Rhinocéros (French, double-spaced, 15k lines)
# Table of Contents appended at end; clean from PERSONNAGES, trim at ToC.
lines = read_file(PLAYS_DIR / "Ionesco_Rhinoceros.txt")
lines = clean(lines,
              start_marker="PERSONNAGES",
              end_marker="Table of Contents",
              norm_spaces=True)
# Remove the Table of Contents line itself
lines = [l for l in lines if "Table of Contents" not in l]
lines = dedupe_blanks(lines)
write_file(OUTPUT_DIR / "Ionesco_Rhinoceros.txt", lines)
print("Done: Ionesco_Rhinoceros.txt")

print("\nAll files written to", OUTPUT_DIR)
