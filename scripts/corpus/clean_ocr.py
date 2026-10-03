"""Clean OCR output files, retaining only actual literary content."""
import re
from pathlib import Path

INPUT_DIR = Path("data/ocr_raw")
OUTPUT_DIR = Path("data/ocr_clean")
OUTPUT_DIR.mkdir(exist_ok=True)


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_file(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def is_ocr_garbage(line):
    """True if line is clearly OCR garbage (mostly non-alphabetic junk)."""
    stripped = line.strip()
    if not stripped:
        return False
    alpha = sum(1 for c in stripped if c.isalpha())
    total = len(stripped)

    # Copyright / special symbols never appear in literary dialogue
    if "©" in stripped or "¥" in stripped:
        return True

    # Very short lines with almost no alphabetic content
    if total <= 4 and alpha <= 1:
        return True

    # Short lines: check meaningful word structure
    real_words = re.findall(r"[a-zA-ZÀ-ÿœŒæÆ]{2,}", stripped)
    if total <= 8:
        if not real_words:
            return True
        # Only tiny words (≤2 chars) and very little alpha total
        if all(len(w) <= 2 for w in real_words) and alpha <= 3:
            return True

    # Longer but no real words at all
    if total > 8 and not real_words:
        return True

    # General high-symbol ratio
    if total > 3 and alpha / total < 0.20:
        return True
    return False


def is_page_header(line, title):
    """True if line is just a running page header — the title surrounded by
    page numbers, OCR symbols, or very short OCR letter artifacts."""
    stripped = line.strip()
    if title.lower() not in stripped.lower():
        return False
    # Remove the title from the line
    remainder = re.sub(re.escape(title), "", stripped, flags=re.IGNORECASE).strip()
    if not remainder:
        return True  # Line is the bare title
    # Check remainder has no substantial text (words of 3+ chars)
    real_words = re.findall(r"[a-zA-ZÀ-ÿœŒæÆ]{3,}", remainder)
    if not real_words:
        return True  # Only numbers / short symbols remain
    # Allow very short OCR artifacts (≤2 chars each, total remainder ≤ 10)
    if all(len(w) <= 2 for w in real_words) and len(remainder) <= 10:
        return True
    return False


def find_first_line(lines, marker, start=0):
    """Return index of first line containing marker (case-sensitive)."""
    for i, line in enumerate(lines[start:], start):
        if marker in line:
            return i
    return start


def remove_running_headers(lines, titles):
    """Remove embedded page-header lines for all given title strings."""
    result = []
    for line in lines:
        if any(is_page_header(line, t) for t in titles):
            continue
        result.append(line)
    return result


def remove_footnotes(lines):
    """Remove editor footnotes: lines like '1. Some explanation...'"""
    result = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d+[.,]\s+[A-ZÀÁÂ]", stripped):
            continue
        result.append(line)
    return result


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
          header_titles=None, remove_garbage=True,
          remove_separators=True, rm_footnotes=False,
          skip_from=None, skip_to=None):
    """Apply all cleaning steps to a list of lines."""
    # Trim to content range
    if start_marker:
        idx = find_first_line(lines, start_marker)
        lines = lines[idx:]
    if end_marker:
        idx = find_first_line(lines, end_marker)
        lines = lines[:idx + 1]  # include the end marker line itself

    # Remove a section between two markers (e.g. COMMENT JOUER)
    if skip_from and skip_to:
        sec_start = find_first_line(lines, skip_from)
        sec_end = find_first_line(lines, skip_to, sec_start)
        lines = lines[:sec_start] + lines[sec_end:]

    # Remove OCR garbage
    if remove_garbage:
        lines = [l for l in lines if not is_ocr_garbage(l)]

    # Remove editor footnotes
    if rm_footnotes:
        lines = remove_footnotes(lines)

    # Remove embedded page / running headers
    if header_titles:
        lines = remove_running_headers(lines, header_titles)

    # Remove pure separator lines
    if remove_separators:
        lines = [l for l in lines
                 if not re.match(r"^[-—_=*]{3,}\s*$", l.strip())]

    lines = dedupe_blanks(lines)
    return lines


# ---------------------------------------------------------------------------
# File-specific processing
# ---------------------------------------------------------------------------

# 1. Brieux — La Femme Seule
# Starts immediately with play content; strip running headers only.
lines = read_file(INPUT_DIR / "Brieux_La Femme Seule.txt")
lines = clean(lines, header_titles=["LA FEMME SEULE"])
write_file(OUTPUT_DIR / "Brieux_La Femme Seule.txt", lines)
print("Done: Brieux_La Femme Seule.txt")


# 2. Brieux — La robe rouge
# First ~285 lines are OCR garbage / publisher front-matter.
# Keep character list (PERSONNAGES → ACTE PREMIER), then play.
lines = read_file(INPUT_DIR / "Brieux_La robe rouge.txt")
pers_start = find_first_line(lines, "PERSONNAGES")
acte_start = find_first_line(lines, "ACTE PREMIER", pers_start)
char_section = [l for l in lines[pers_start:acte_start] if not is_ocr_garbage(l)]
play_section = clean(lines[acte_start:], header_titles=["LA ROBE ROUGE", "BRIEUX"])
lines = char_section + ["\n"] + play_section
lines = dedupe_blanks(lines)
write_file(OUTPUT_DIR / "Brieux_La robe rouge.txt", lines)
print("Done: Brieux_La robe rouge.txt")


# 3. Dumas — Camille
# Library catalog + OCR junk precedes actual content.
# Keep character list, skip OCR garbage block, then start at ACT FIRST.
lines = read_file(INPUT_DIR / "Dumas_Camille.txt")
char_start = find_first_line(lines, "CHARACTERS IN THE PLAY.")
act_start = find_first_line(lines, "ACT FIRST.")
char_section = [l for l in lines[char_start:act_start] if not is_ocr_garbage(l)]
play_section = clean(lines[act_start:], header_titles=["CAMILLE"])
lines = char_section + ["\n"] + play_section
lines = dedupe_blanks(lines)
write_file(OUTPUT_DIR / "Dumas_Camille.txt", lines)
print("Done: Dumas_Camille.txt")


# 4. Dumas — The Demi monde
# Internet Archive header + publisher / translator note at top.
# "CAST OF CHARACTERS," marks start of literary content.
lines = read_file(INPUT_DIR / "Dumas_The Demi monde.txt")
lines = clean(lines, start_marker="CAST OF CHARACTERS,",
              header_titles=["DEMI-MONDE", "TRANSLATOR'S NOTE"])
write_file(OUTPUT_DIR / "Dumas_The Demi monde.txt", lines)
print("Done: Dumas_The Demi monde.txt")


# 5. Genet — Le balcon
# PERSONNAGES is the start of literary content.
# COMMENT JOUER (director's notes) runs from that header to PREMIER TABLEAU.
lines = read_file(INPUT_DIR / "Genet_Le balcon.txt")
lines = clean(lines, start_marker="PERSONNAGES",
              skip_from="COMMENT JOUER", skip_to="PREMIER TABLEAU",
              header_titles=["LE BALCON"])
lines = [l for l in lines if not l.strip().startswith("© Marc")]
lines = dedupe_blanks(lines)
write_file(OUTPUT_DIR / "Genet_Le balcon.txt", lines)
print("Done: Genet_Le balcon.txt")


# 6. Genet — Les bonnes
# COMMENT JOUER section precedes the actual play.
# "La chambre de Madame." is the first stage direction of the play.
lines = read_file(INPUT_DIR / "Genet_Les bonnes.txt")
lines = clean(lines, start_marker="La chambre de Madame.",
              header_titles=["Les Bonnes", "Comment jouer"])
write_file(OUTPUT_DIR / "Genet_Les bonnes.txt", lines)
print("Done: Genet_Les bonnes.txt")


# 7. Ionesco — La cantatrice chauve
# Academic edition with interleaved footnotes and analysis.
# Content: from "Personnages" to "RIDEAU" (inclusive).
lines = read_file(INPUT_DIR / "Ionesco_La cantatrice chauve.txt")
lines = clean(lines, start_marker="Personnages", end_marker="RIDEAU",
              rm_footnotes=True,
              header_titles=["La Cantatrice chauve", "Cantatrice chauve"])
lines = [l for l in lines
         if "Digitized by the Internet Archive" not in l
         and "https://archive.org" not in l
         and l.strip() != "in 2024"]
lines = dedupe_blanks(lines)
write_file(OUTPUT_DIR / "Ionesco_La cantatrice chauve.txt", lines)
print("Done: Ionesco_La cantatrice chauve.txt")


# 8. Ionesco — Rhinoceros
# Publisher info at top; CHARACTERS marks start of literary content.
lines = read_file(INPUT_DIR / "Ionesco_Rhinoceros.txt")
lines = clean(lines, start_marker="CHARACTERS",
              header_titles=["RHINOCEROS", "IONESCO",
                             "THE FUTURE IS IN EGGS", "THE LEADER"])
write_file(OUTPUT_DIR / "Ionesco_Rhinoceros.txt", lines)
print("Done: Ionesco_Rhinoceros.txt")

print("\nAll files written to", OUTPUT_DIR)
