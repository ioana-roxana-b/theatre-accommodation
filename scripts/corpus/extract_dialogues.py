"""
extract_dialogues.py
Extracts speaker/dialogue data from cleaned play texts and writes
character_info.csv, quotation_info.csv, and play_text.txt per play,
matching the PDNC CSV schema.
"""

import re, csv, json
from pathlib import Path
from collections import defaultdict, Counter

CLEAN_PLAYS  = Path("data/plays_clean")
CLEAN_OUTPUT = Path("data/ocr_clean")
CLEAN_OCR    = CLEAN_OUTPUT  # alias kept for clarity
CORPUS       = Path("data/plays_dialogue")

# ── tiny helpers ─────────────────────────────────────────────────────────

def read_text(p):
    return Path(p).read_text(encoding="utf-8")

def norm_sp(s):
    """Normalize a speaker label: strip trailing punctuation/spaces."""
    s = re.sub(r'[\s.,;:!?\-—]+$', '', s.strip())
    return re.sub(r'\s{2,}', ' ', s).strip()

def gender_from_name(name):
    n = name.lower()
    if any(t in n for t in [
        'madame','mme.','mme ','mrs.','mrs ','miss ','miss.','countess',
        'baronne','vicomtesse','wife','mother','lady','sister','woman',
        'la femme','la reine','la bonne','solange','claire','madame',
    ]):
        return 'F'
    if any(t in n for t in [
        'monsieur','m.','mr.','mr ','sir ','lord ','baron','le général',
        'docteur','dr.','colonel','captain','le juge','le bourreau',
        'le chef','le patron','l\'évêque','le vieux','le mendiant',
    ]):
        return 'M'
    return 'U'

def category_by_rank(name, counts):
    ranked = sorted(counts, key=counts.get, reverse=True)
    n = len(ranked)
    if name not in ranked:
        return 'minor'
    rank = ranked.index(name)
    if rank < max(1, round(n * 0.2)):
        return 'major'
    if rank < max(2, round(n * 0.5)):
        return 'intermediate'
    return 'minor'

# Non-speaker ALL-CAPS patterns to skip
_SKIP = re.compile(
    r'^(ACTE?|SCENE|SCÈNE|TABLEAU|PROLOGUE|ÉPILOGUE|EPILOGUE|'
    r'RIDEAU|FIN\b|PERSONNAGES|CHARACTERS|CAST|THE END|CURTAIN|'
    r'PAUSE|SILENCE|APPENDIX|DÉCOR|DECOR|ACT\s+[IVX\d]|'
    r'SCENE\s+[IVX\d\w]|PREMIÈRE|PREMIER|DEUXIÈME|TROISIÈME|'
    r'QUATRIÈME|CINQUIÈME|SIXIÈME|SIXIEME|SEPTIÈME|SEPTIEME|'
    r'HUITIÈME|HUITIEME|NEUVIÈME|NEUVIEME|DIXIÈME|DIXIEME|'
    r'FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH)',
    re.IGNORECASE,
)

# Regex to strip trailing scene/act headings that bleed into dialogue text.
# Matches patterns like "... J'y serai. SCÈNE DERNIÈRE" or "... ACTE II" at end of text.
_TRAILING_HEADING = re.compile(
    r'[\s\-—:"“”]+(?:[A-Z0-9\.\,\-—:"“”]*\s*)?'
    r'(?:ACTE?|SC[EÈ]NE|SCÈNE|TABLEAU|ACT|SCENE)\s+\S.*$',
    re.IGNORECASE,
)

# Common French / English words that look like ALL-CAPS speaker lines but aren't names.
# Also includes theater stage terms that appear on their own line between speeches.
_COMMON_WORDS = {
    # French exclamations / filler words
    'adieu','ainsi','allez','allons','alors','ah','apres','après',
    'attention','bien','bonsoir','bonjour','bravo',
    'calmez','calmez-vous','cependant','comment','courage',
    'decidement','devinez','donc','dors',
    'ecoutez','ecoute','elle','ensemble','ensuite','enfin',
    'jamais','justement','lui','mais','maintenant',
    'merci','moi','monsieur','non','oui','pardon','parlez','patience',
    'peut','pourquoi','precisement','regarde','repose','rien','silence','si','soit',
    'tiens','tout','toutes','tous','voila','voi','voici','voila','voyons',
    'vous','votre','vrai','vraiment','parfaitement','mon dieu','dieu',
    # French theater stage terms (appear on a line alone between scenes)
    'les memes','la meme','les mêmes','la même',
    'ensemble','choeur','chœur',
    # English
    'yes','no','oh','ah','well','now','come','go','stop','wait','please',
    'hello','goodbye','pardon','indeed','quite','never','always','together','all',
}

def _is_skip(s):
    stripped = s.strip()
    if _SKIP.match(stripped):
        return True
    # Strip trailing punctuation and check against common-words blacklist.
    # Also strip diacritics for the comparison so accented variants are caught.
    import unicodedata as _ud
    word = re.sub(r'[\.\!\?\,\;]+$', '', stripped).strip().lower()
    word_nodiac = ''.join(
        c for c in _ud.normalize('NFD', word) if _ud.category(c) != 'Mn'
    )
    if word in _COMMON_WORDS or word_nodiac in _COMMON_WORDS:
        return True
    return False

def _alpha_upper_ratio(s):
    alpha = [c for c in s if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)

def _name_part(s):
    """Return the name part of a speaker line (before comma, paren, or '. lowercase').

    The dot-lowercase rule handles French format "THÉRÈSE. anrieuse." → "THÉRÈSE".
    The 2-char uppercase lookbehind prevents title abbreviations like "M." or "Dr."
    from being stripped (they have fewer than 2 consecutive uppercase chars before the dot).
    """
    s0 = re.split(r'[,\(]', s)[0].strip()
    # French format: "THÉRÈSE. anrieuse." → "THÉRÈSE"
    # Lookbehind requires 2 consecutive uppercase chars before the dot so that
    # single-letter titles like "M." or "Mme." are not split.
    s0 = re.sub(
        r'(?<=[A-ZÀÁÂÉÈÊËÙÚÛÜÇŒÆ]{2})\.\s+[a-zàáâéèêëîïùúûüçœæ].*$',
        '', s0,
    ).strip()
    return s0

def looks_caps_speaker(line):
    """True if line looks like an ALL-CAPS speaker label."""
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if _is_skip(s):
        return False
    np = _name_part(s)
    if not np or len(np) > 50:
        return False
    # Remove trailing dot
    np = re.sub(r'\.$', '', np).strip()
    alpha = [c for c in np if c.isalpha()]
    if not alpha:
        return False
    return _alpha_upper_ratio(np) >= 0.80 and len(np.split()) <= 7

def looks_titlecase_speaker(line):
    """True if line looks like a Title-Case speaker label (French plays)."""
    s = line.strip()
    if not s or len(s) > 70:
        return False
    if _is_skip(s):
        return False
    np = re.sub(r'\.$', '', _name_part(s)).strip()
    if not np:
        return False
    words = np.split()
    if not words or len(words) > 5:
        return False
    alpha_words = [w for w in words if w[0].isalpha()]
    return bool(alpha_words) and all(w[0].isupper() for w in alpha_words)

def clean_play_text(text):
    """Clean OCR noise from raw play text before saving play_text.txt.

    Applied to the full play text; the cleaned version is what gets saved and
    what byte spans reference, so parsers and span-finders see the same text.
    """
    # Normalize Windows line endings first so all patterns below use \n only
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Strip trailing whitespace BEFORE the hyphen join, otherwise "gau- \nche"
    # (hyphen + trailing space) would only reveal the break after the join step.
    text = re.sub(r'[ \t]+\n', '\n', text)

    # Join OCR hyphenated line breaks across one or more blank lines, but ONLY
    # when the continuation starts with a lowercase letter — this prevents joining
    # "seri-\n\nCAMILLE" where CAMILLE is a new speaker label, not a word part.
    text = re.sub(r'(\w)-\n+(?=[a-záàâéèêëîïùúûüçœæô])', r'\1', text)

    # Remove pipe | characters (OCR page-margin / column-rule artifacts):
    #   "MADAME GUÉRET |"  →  "MADAME GUÉRET"
    # Use [ \t] not \s to avoid consuming newlines (which would join lines).
    text = re.sub(r'[ \t]*\|[ \t]*', ' ', text)

    # Remove leading hyphen or colon before ALL-CAPS speaker name (OCR artifacts):
    #   "- MADAME GUÉRET"  →  "MADAME GUÉRET"
    #   ": RENÉ"           →  "RENÉ"
    # Safe: requires 2+ consecutive uppercase letters so "- Oui" is untouched.
    text = re.sub(r'(?m)^[-:]\s+(?=[A-ZÀÁÂÉÈÊËÙÚÛÜÇŒÆ]{2})', '', text)

    # Remove single leading uppercase letter + space before ALL-CAPS speaker names:
    #   "L THÉRÈSE"  →  "THÉRÈSE"   (single OCR-noise letter before name)
    # The lookahead requires 2+ uppercase chars so "I think" is untouched.
    text = re.sub(r'(?m)^[A-ZÀÁÂÉÈÊËÙÚÛÜÇŒÆ]\s+(?=[A-ZÀÁÂÉÈÊËÙÚÛÜÇŒÆ]{2})', '', text)

    # Remove leading bracket before ALL-CAPS speaker names (OCR bracket noise):
    #   "] THÉRÈSE"  →  "THÉRÈSE"
    text = re.sub(r'(?m)^\]\s+(?=[A-ZÀÁÂÉÈÊËÙÚÛÜÇŒÆ]{2})', '', text)

    # Remove leading dot(s) + horizontal space at start of line (orphaned period noise):
    #   ". Et enfin"  →  "Et enfin"
    # Use [ \t] not \s to avoid consuming the trailing newline and merging lines.
    text = re.sub(r'(?m)^\.[ \t]+', '', text)

    # Remove leading underscore(s) at start of line (OCR underline noise):
    #   "_ succès"  →  "succès"
    text = re.sub(r'(?m)^_+[ \t]*', '', text)

    # Period immediately followed by Unicode ellipsis is OCR trailing noise:
    #   "extraordinaire.…."  →  "extraordinaire."
    text = re.sub(r'\.[….]*…[.…]*', '.', text)
    # Normalize any remaining standalone Unicode ellipsis to ASCII "..."
    text = text.replace('…', '...')
    # Collapse 4+ consecutive ASCII dots to standard 3-dot ellipsis
    text = re.sub(r'\.{4,}', '...', text)

    # Remove OCR-corrupted ACT/SCENE page-header lines (standalone running headers):
    #   "AOT 1.—SCENE I. > 9"   (OCR of "ACT I.—SCENE I." + page number)
    #   "i ACT J.—SCENE I. 15"  (leading lowercase "i" is OCR noise)
    #   "; ACT 1.—SCENE I. 13"  (leading semicolon is OCR noise)
    # Pattern: optional leading noise ([space/tab/;] and single i/j/I/J),
    # then an ACT-like word (including OCR variants AOT, ACI), then SCENE later on the line.
    text = re.sub(
        r'(?m)^[ \t;]*[iIjJ]?\s*(?:ACT|AOT|ACI)[^\n]*\bSCENE\b[^\n]*$',
        '',
        text,
    )

    # Fix Becque-style page-header lines that embed a speaker name at the end:
    #   "et "6 THE I VULTURES [act u ee oe ee eae MARIE"
    #   →  "\nMARIE"   (speaker moves to its own line so the parser sees the boundary)
    # Matches: long line containing "[act " (OCR bracket-act noise) ending with
    # one or more ALL-CAPS words (the speaker name).
    text = re.sub(
        r'(?m)^.{10,}\[act\s[^\n]+?([A-Z]{3,}(?:\s+[A-Z]+)*)\s*$',
        r'\n\1',
        text,
    )

    # Strip trailing whitespace from each line again to clean up any spaces introduced
    # by the pipe and other removals above.
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'[ \t]+$', '', text)
    return text


def strip_inline_stage(text):
    """Remove parenthetical and bracketed stage directions from dialogue text."""
    text = re.sub(r'\([^)]{1,200}\)', '', text)
    text = re.sub(r'\[[^\]]{1,300}\]', '', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def clean_stage_dirs(text):
    """Strip all parenthetical/bracketed stage directions from dialogue text,
    then tidy up leading punctuation artifacts and extra whitespace."""
    # Remove (stage directions) of any length
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove [stage directions] of any length
    text = re.sub(r'\[[^\]]*\]', '', text)
    # Strip trailing scene/act headings that bled into dialogue text:
    #   "J'y serai, merci. SCÈNE DERNIÈRE"  →  "J'y serai, merci."
    #   "She smiled. ACT II SCENE I"        →  "She smiled."
    text = _TRAILING_HEADING.sub('', text)
    # Remove orphaned period after sentence-ending punctuation:
    #   "angry? . Forgive" → "angry? Forgive"  (period left after removing stage dir)
    text = re.sub(r'([?!,;])\s*\.\s+', r'\1 ', text)
    # Remove orphaned space-dot-space in the middle:
    #   "came . He said" → "came He said"
    text = re.sub(r'\s+\.\s+', ' ', text)
    # Remove leading punctuation/whitespace artifacts (e.g. ". " or "— " or "[ ")
    text = re.sub(r'^[\s\.\,\;\:\!\?\-—\[\]]+', '', text)
    # Remove trailing standalone punctuation/whitespace that is not sentence-ending
    # (e.g. trailing " ;" or " ." left after removing stage directions or OCR noise).
    # We keep trailing ? and ! as they are legitimate sentence-enders.
    text = re.sub(r'[\s\,\;\:\-—\[\]]+$', '', text)
    # Collapse internal runs of whitespace/newlines to single space
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def find_byte_span(tb, segment, start=0):
    """Find (start_byte, end_byte) of segment in text_bytes tb, starting at start.
    Returns (span_or_None, new_start).

    Uses whitespace-flexible regex search first (handles cases where dialogue was
    collected by joining lines with spaces, while play_text.txt preserves newlines).
    Falls back to exact byte search for single-word segments where regex is not useful.
    """
    words = segment.split()

    if len(words) >= 2:
        # Flex search: match words in sequence, any whitespace allowed between them.
        # This correctly handles ". Forgive me." matching ".\n\nForgive me." in the file.
        text_suffix = tb[start:].decode('utf-8', errors='replace')
        pattern = r'\s+'.join(re.escape(w) for w in words[:80])
        m = re.search(pattern, text_suffix)
        if not m:
            return None, start
        abs_byte_start = start + len(text_suffix[:m.start()].encode('utf-8'))
        abs_byte_end   = start + len(text_suffix[:m.end()].encode('utf-8'))
        return (abs_byte_start, abs_byte_end), abs_byte_end

    # Exact match fallback for single-word (or empty) segments
    seg = segment.encode('utf-8')
    pos = tb.find(seg, start)
    if pos == -1:
        return None, start
    return (pos, pos + len(seg)), pos + len(seg)


# ── FORMAT PARSERS ───────────────────────────────────────────────────────
# Each returns list of (raw_label, dialogue_text) pairs.
# raw_label  = the speaker header string as it appears in the text
# dialogue_text = the spoken text (may include inline [stage] fragments)

# ── 1. fr_emdash: SPEAKER (stage). — dialogue   (same line) ─────────────
# Covers: Beckett FR, Ionesco cantatrice (uses '-' instead of '—')
_EMDASH_RE = re.compile(
    r'^([A-ZÀÁÂÉÈÊËÙÚÛÜÇŒÆ\'\-][A-ZÀÁÂÉÈÊËÙÚÛÜÇŒÆa-z\s\'\-\.]*?)'  # name
    r'(?:\s*[,\(][^)]*\))?'                                             # opt stage
    r'\s*[\.\,]?\s*[—\-]\s+'                                           # . — or . -
    r'(.+)$',                                                           # dialogue
    re.UNICODE,
)

def parse_fr_emdash(text, known_speakers=None):
    known = None
    if known_speakers:
        known = {_ocr_key(s) for s in known_speakers}
    results = []
    for line in text.splitlines():
        raw = line.strip()
        m = _EMDASH_RE.match(raw)
        if not m:
            # Strip leading OCR noise (digits, symbols, lowercase tokens) before
            # the first uppercase letter so lines like "135 M. MARTIN. — Par!" or
            # "ao )~ Mme SMITH. — Ci!" are still matched.
            noise_stripped = re.sub(r'^[^A-ZÀÁÂÉÈÊËÙÚÛÜÇŒÆ]+', '', raw)
            if noise_stripped and noise_stripped != raw:
                m = _EMDASH_RE.match(noise_stripped)
        if not m:
            continue
        speaker = norm_sp(m.group(1))
        if _is_skip(speaker):
            continue
        if known is not None:
            sk = _ocr_key(speaker)
            # Accept if within edit-distance 2 of any known speaker
            if not any(_lev(sk, ks) <= 2 for ks in known):
                continue
        dialogue = m.group(2).strip()
        parts = re.split(r'\([^)]*\)\s*\.?\s*[—\-]\s*', dialogue)
        full = ' '.join(p.strip() for p in parts if p.strip())
        if speaker and full:
            results.append((speaker, full))
    return results


# ── 2. en_colon_inline: SPEAKER: [stage] dialogue   (same line) ──────────
# Covers: Beckett Endgame, Ionesco Bald Soprano, Ionesco Rhinoceros EN
_COLON_INLINE_RE = re.compile(
    r'^([A-Z][A-Z\s\.\'\-]*?)\s*:\s*(?:\[[^\]]*\]\s*)?(.+)$',
    re.UNICODE,
)

def parse_en_colon_inline(text):
    results = []
    for line in text.splitlines():
        s = line.strip()
        m = _COLON_INLINE_RE.match(s)
        if not m:
            continue
        speaker = norm_sp(m.group(1))
        dialogue = m.group(2).strip()
        # Skip stage-direction-only "dialogue" like [continues to read]
        if re.match(r'^\[.*\]$', dialogue):
            continue
        if speaker and dialogue and not _is_skip(speaker) and looks_caps_speaker(m.group(1)):
            results.append((speaker, dialogue))
    return results


# ── 3. en_colon_own: SPEAKER:\n(blank)\ndialogue ─────────────────────────
# Covers: Beckett Waiting for Godot, Genet Balcony, Genet The Maids
def parse_en_colon_own(text):
    results = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].rstrip()
        # Speaker line ends with ':'  (possibly with trailing spaces/stage dir)
        m = re.match(r'^([A-Z][A-Z\s\.\'\-]*?)\s*(?:\[[^\]]*\])?\s*:\s*$', s)
        if m and not _is_skip(m.group(1)):
            speaker = norm_sp(m.group(1))
            i += 1
            # Collect dialogue lines until next speaker or blank boundary
            seg_lines = []
            while i < len(lines):
                sl = lines[i].rstrip()
                # Stop at structural headings (SCÈNE, ACTE, etc.)
                if _SKIP.match(sl.strip()):
                    break
                # Stop if we hit a new speaker line
                if re.match(r'^([A-Z][A-Z\s\.\'\-]*?)\s*(?:\[[^\]]*\])?\s*:\s*$', sl) and not _is_skip(sl):
                    break
                seg_lines.append(sl)
                i += 1
            dialogue = ' '.join(l for l in seg_lines if l.strip())
            dialogue = re.sub(r'\s{2,}', ' ', dialogue).strip()
            if speaker and dialogue:
                results.append((speaker, dialogue))
        else:
            i += 1
    return results


# ── 4. en_caps_own: ALL-CAPS speaker on own line, dialogue below ──────────
# Covers: Becque The Vultures, Becque Woman of Paris
def parse_en_caps_own(text, min_count=2):
    lines = text.splitlines()
    # Two-pass: build known speaker set
    counts = Counter()
    for line in lines:
        s = line.strip()
        if looks_caps_speaker(s) and not re.search(r'[a-z]{3}', s) and not _is_skip(s):
            key = re.sub(r'\s*\(.*\)$', '', norm_sp(_name_part(s))).strip().upper()
            if key:
                counts[key] += 1
    canon = {k for k, v in counts.items() if v >= min_count}

    results = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        key = re.sub(r'\s*\(.*\)$', '', norm_sp(_name_part(s))).strip().upper()
        if key in canon and looks_caps_speaker(s) and not re.search(r'[a-z]{3}', s):
            speaker = re.sub(r'\s*\(.*\)$', '', norm_sp(_name_part(s))).strip()
            i += 1
            seg_lines = []
            while i < len(lines):
                sl = lines[i].strip()
                if not sl:
                    i += 1
                    continue
                # Stop at structural headings (SCÈNE, ACTE, etc.)
                if _SKIP.match(sl):
                    break
                sl_key = re.sub(r'\s*\(.*\)$', '', norm_sp(_name_part(sl))).strip().upper()
                if sl_key in canon and looks_caps_speaker(sl) and not re.search(r'[a-z]{3}', sl):
                    break
                seg_lines.append(sl)
                i += 1
            dialogue = ' '.join(seg_lines)
            dialogue = re.sub(r'\s{2,}', ' ', dialogue).strip()
            if speaker and dialogue and not _is_skip(speaker):
                results.append((speaker, dialogue))
        else:
            i += 1
    return results


# ── 5. en_dot_inline: Speaker. dialogue  (Title-Case, same line) ─────────
# Covers: Dumas Camille EN, Brieux Woman on Her Own, Dumas Demi monde EN
_DOT_INLINE_RE = re.compile(
    r'^([A-Z][a-zA-ZÀ-ÿœŒæÆ\s\'\.\-]{1,40}?)\s*[\.\!]\s*(?:\[([^\]]*)\]\s*)?(.+)$',
    re.UNICODE,
)

_STAGE_PREFIXES = re.compile(
    r'^(Enter|Exit|Exeunt|Re-enter|Re-exit|Aside|Stage|Scene|Act)\b',
    re.IGNORECASE,
)

def parse_en_dot_inline(text):
    results = []
    for line in text.splitlines():
        s = line.strip()
        m = _DOT_INLINE_RE.match(s)
        if not m:
            continue
        speaker = norm_sp(m.group(1))
        dialogue = m.group(3).strip()
        # Skip stage-direction speaker labels like "Enter Mondoubleau. dialogue"
        if _STAGE_PREFIXES.match(speaker):
            continue
        # Skip if speaker has lowercase-starting words (likely prose)
        words = speaker.split()
        if not all(w[0].isupper() for w in words if w and w[0].isalpha()):
            continue
        if len(words) > 5 or _is_skip(speaker):
            continue
        if speaker and dialogue:
            results.append((speaker, dialogue))
    return results


# ── 6. fr_caps_own: CAPS/TitleCase speaker own line, dialogue below ───────
# Covers: Becque La Parisienne, Dumas FR plays, Ionesco Rhinoceros FR,
#         Brieux La Femme Seule, Genet Les Bonnes, Genet Le Balcon
#
# Two-pass: first collect candidate names, keep only those appearing >= min_count
# times as candidate speaker lines (eliminates standalone French exclamations like
# "Adieu." that look like speaker labels but appear only once or twice).
def _candidate_speakers(lines, allow_titlecase, min_count):
    counts = Counter()
    for line in lines:
        s = line.strip()
        if not s or _is_skip(s):
            continue
        if looks_caps_speaker(s) or (allow_titlecase and looks_titlecase_speaker(s)):
            key = re.sub(r'\.$', '', norm_sp(_name_part(s))).strip().upper()
            # Also reject keys that are themselves common words, even when the
            # full line was not caught (e.g. "OUI, madame." → key "OUI").
            if key and not _is_skip(key):
                counts[key] += 1
    return {k for k, v in counts.items() if v >= min_count}

def _normalise_spkr(s):
    return re.sub(r'\.$', '', norm_sp(_name_part(s))).strip()

def parse_fr_caps_own(text, allow_titlecase=True, known_speakers=None, min_count=3,
                      content_start=None):
    lines = text.splitlines()
    # Find starting line index
    start_idx = 0
    if content_start:
        for li, ln in enumerate(lines):
            if content_start in ln:
                start_idx = li
                break

    if known_speakers is not None:
        canon = {n.upper() for n in known_speakers}
        canon_extended = canon
    else:
        canon = _candidate_speakers(lines[start_idx:], allow_titlecase, min_count)
        # Extended canon: all candidates (count ≥ 1) used only for BREAKING collection.
        # This prevents rare character names (below min_count) from being swallowed into
        # a previous speaker's dialogue.
        # Extended canon: ALL-CAPS-only candidates (count ≥ 2) used for both
        # starting and breaking. Title-case candidates excluded (avoids false
        # positives like "Tais-toi."). min_count=2 filters single-occurrence
        # OCR noise while still catching rare characters like MAUD/ANTOINETTE.
        canon_extended = _candidate_speakers(lines[start_idx:], False, 2)

    # Pre-build OCR-key sets for fuzzy matching of corrupted speaker names.
    canon_ocr = {_ocr_key(c) for c in canon}
    canon_extended_ocr = {_ocr_key(c) for c in canon_extended}

    def _in_canon(key, original_line):
        """Exact canon membership OR fuzzy OCR match for low-frequency OCR variants.
        original_line is used for looks_caps_speaker so we check the UNMODIFIED line,
        not the already-uppercased key (which would always pass the uppercase ratio test)."""
        return key in canon or (
            looks_caps_speaker(original_line) and
            _fuzzy_in_canon(_ocr_key(key), canon_ocr)
        )

    def _in_canon_extended(key, original_line):
        """ALL-CAPS-only extended canon (count ≥ 1).
        Used for both starting and breaking, to catch rare character names
        that fall below the min_count threshold for the main canon."""
        return key in canon_extended or (
            looks_caps_speaker(original_line) and
            _fuzzy_in_canon(_ocr_key(key), canon_extended_ocr)
        )

    results = []
    i = start_idx
    while i < len(lines):
        s = lines[i].strip()
        key = _normalise_spkr(s).upper()
        if _in_canon(key, s) and not _is_skip(s) and not _is_skip(key):
            speaker = _normalise_spkr(s)
            i += 1
            seg_lines = []
            while i < len(lines):
                sl = lines[i].strip()
                if not sl:
                    i += 1
                    continue
                # Stop at a structural heading (SCÈNE, ACTE, TABLEAU …)
                if _SKIP.match(sl):
                    break
                sl_key = _normalise_spkr(sl).upper()
                if _in_canon(sl_key, sl) and not _is_skip(sl) and not _is_skip(sl_key):
                    break
                seg_lines.append(sl)
                i += 1
            dialogue = ' '.join(seg_lines)
            dialogue = re.sub(r'\s{2,}', ' ', dialogue).strip()
            if speaker and dialogue:
                results.append((speaker, dialogue))
        else:
            i += 1
    return results


# ── 7. fr_titlecase_emdash: Speaker. — dialogue (Title Case) ─────────────
# Covers: Becque Les Corbeaux
_TC_EMDASH_RE = re.compile(
    r'^([A-ZÀÁÂ][a-zàáâéèêëùúûüçœæA-Z\s\'\-\.]{1,50}?)\s*[,\.]?\s*[—\-]\s+(.+)$',
    re.UNICODE,
)

def parse_fr_titlecase_emdash(text):
    results = []
    for line in text.splitlines():
        m = _TC_EMDASH_RE.match(line.strip())
        if not m:
            continue
        speaker = norm_sp(m.group(1))
        dialogue = m.group(2).strip()
        words = speaker.split()
        if len(words) > 5 or _is_skip(speaker):
            continue
        if speaker and dialogue:
            results.append((speaker, dialogue))
    return results


# ── POST-PROCESSING: build character list & normalize names ──────────────

def _ocr_key(name):
    """Aggressive normalization for OCR variant matching.
    Strips accents/diacritics, lowercases, removes non-alpha, collapses spaces."""
    import unicodedata
    s = unicodedata.normalize('NFD', name)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')  # drop combining chars
    s = re.sub(r'[^a-zA-Z\s]', '', s).lower().strip()
    s = re.sub(r'\s+', ' ', s)
    # Common OCR confusions: rn→m, li→u, vv→w, etc.
    s = s.replace('rn', 'm').replace('iii', 'm').replace('ii', 'n')
    return s

def _lev(a, b):
    """Levenshtein distance (fast for short strings)."""
    if a == b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j] + (ca != cb), prev[j+1] + 1, curr[j] + 1))
        prev = curr
    return prev[-1]

def _fuzzy_in_canon(ocr_key, canon_ocr_set):
    """True if ocr_key (already _ocr_key-normalised) is an exact or Levenshtein-close
    match for any member of canon_ocr_set (pre-built as {_ocr_key(c) for c in canon}).
    Uses the same 2-char prefix guard and distance thresholds as build_char_map."""
    if ocr_key in canon_ocr_set:
        return True
    for c_ocr in canon_ocr_set:
        if ocr_key[:2] != c_ocr[:2]:
            continue
        min_len = min(len(ocr_key), len(c_ocr))
        max_dist = 2 if min_len <= 8 else 3
        if _lev(ocr_key, c_ocr) <= max_dist:
            return True
    return False

def build_char_map(utterances):
    """Return (canon_map, alias_map).
    Merges OCR variants using OCR-normalised key + Levenshtein ≤ 2."""
    raw_labels = list(dict.fromkeys(raw for raw, _ in utterances))
    freq = Counter(raw for raw, _ in utterances)

    # Union-Find for clustering
    parent = {r: r for r in raw_labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Canonical = whichever has higher frequency
        if freq.get(ra, 0) >= freq.get(rb, 0):
            parent[rb] = ra
        else:
            parent[ra] = rb

    # Cluster similar names: use Levenshtein on OCR-normalised keys.
    # Threshold scales with name length to avoid merging short unrelated names.
    indexed = [(r, _ocr_key(r)) for r in raw_labels]
    for i, (r1, k1) in enumerate(indexed):
        for r2, k2 in indexed[i+1:]:
            if r1 == r2:
                continue
            min_len = min(len(k1), len(k2))
            if min_len < 3:
                continue
            # Require shared first 2 chars (prevents merging totally different names)
            if k1[:2] != k2[:2]:
                continue
            max_dist = 2 if min_len <= 8 else 3
            if _lev(k1, k2) <= max_dist:
                union(r1, r2)

    # Build maps
    alias_map = defaultdict(set)
    for r in raw_labels:
        alias_map[find(r)].add(r)

    canon_map = {raw: find(raw) for raw, _ in utterances}
    return canon_map, alias_map


def compute_addressees(utterances, canon_map, alias_map, speech_counts, window=10):
    """Return list of addressee lists, one per utterance.

    For each utterance i the addressees are the canonical Main Names of all other
    speakers found within ±window utterances (scene co-presence heuristic).
    The Main Name is the most-frequent alias of each canonical cluster.
    """
    # Build canonical -> main_name lookup
    canon_to_main = {
        canon: max(aliases, key=lambda a: speech_counts.get(a, 0))
        for canon, aliases in alias_map.items()
    }

    n = len(utterances)
    result = []
    for i, (raw, _) in enumerate(utterances):
        current_canon = canon_map.get(raw, raw)
        present = set()
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j == i:
                continue
            other_raw, _ = utterances[j]
            other_canon = canon_map.get(other_raw, other_raw)
            if other_canon != current_canon:
                present.add(canon_to_main.get(other_canon, other_canon))
        result.append(sorted(present))
    return result


def assign_ids(alias_map, speech_counts):
    """Assign integer IDs sorted by name, return list of char dicts.
    Uses the most-frequent alias as Main Name (cleaner form preferred)."""
    chars = []
    for i, (canon, aliases) in enumerate(sorted(alias_map.items())):
        # Pick most-frequent alias as display name (usually the cleanest form)
        best = max(aliases, key=lambda a: speech_counts.get(a, 0))
        chars.append({
            'Character ID': i,
            'Main Name': best,
            'Aliases': aliases,
            'Gender': gender_from_name(best),
            'Category': category_by_rank(canon, speech_counts),
        })
    return chars


# ── OUTPUT WRITERS ────────────────────────────────────────────────────────

def write_character_csv(path, chars):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['Character ID','Main Name','Aliases','Gender','Category'])
        w.writeheader()
        for c in chars:
            w.writerow({
                'Character ID': c['Character ID'],
                'Main Name': c['Main Name'],
                'Aliases': str(c['Aliases']),
                'Gender': c['Gender'],
                'Category': c['Category'],
            })


def write_quotation_csv(path, utterances, canon_map, play_text_bytes, addressees_list):
    fieldnames = [
        'quoteID','quoteText','subQuotationList','quoteByteSpans',
        'speaker','addressees','quoteType','referringExpression',
    ]
    offset = 0
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for idx, (raw_label, dialogue) in enumerate(utterances):
            # Skip utterances that are entirely stage directions (nothing left after clean)
            if not clean_stage_dirs(dialogue):
                continue
            canon = canon_map.get(raw_label, raw_label)
            addressees = addressees_list[idx] if idx < len(addressees_list) else []
            # Split dialogue into sub-quotations at stage-direction paragraphs
            # (stage dirs are in [brackets] or (parens))
            parts = re.split(r'\s*[\[\(][^\]\)]{5,200}[\]\)]\s*', dialogue)
            parts = [p.strip() for p in parts if p.strip()]
            if not parts:
                parts = [dialogue]

            byte_spans = []
            actual_parts = []
            for part in parts:
                span, offset = find_byte_span(play_text_bytes, part, offset)
                if span:
                    byte_spans.append(list(span))
                    actual_parts.append(
                        play_text_bytes[span[0]:span[1]].decode('utf-8', errors='replace')
                    )
                else:
                    byte_spans.append([])
                    actual_parts.append(part)

            # Clean stage directions from stored text (byte spans still point
            # to the raw file positions, which is intentional).
            clean_parts = [clean_stage_dirs(p) for p in actual_parts]
            # Drop parts that are empty after cleaning (pure stage-direction fragments)
            kept = [(cp, sp) for cp, sp in zip(clean_parts, byte_spans) if cp]
            if kept:
                clean_parts, byte_spans = zip(*kept)
                clean_parts = list(clean_parts)
                byte_spans  = list(byte_spans)
            else:
                clean_parts = [clean_stage_dirs(dialogue)] or [dialogue]
                byte_spans  = [[]]

            w.writerow({
                'quoteID': f'Q{idx}',
                'quoteText': clean_stage_dirs(dialogue),
                'subQuotationList': json.dumps(clean_parts, ensure_ascii=False),
                'quoteByteSpans': json.dumps(byte_spans),
                'speaker': canon,
                'addressees': json.dumps(addressees, ensure_ascii=False),
                'quoteType': 'Explicit',
                'referringExpression': raw_label,
            })


# ── MAIN PROCESSING ───────────────────────────────────────────────────────

def process_play(input_path, output_dir_name, parser_fn, parser_kwargs=None):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"  SKIP (not found): {input_path}")
        return
    parser_kwargs = parser_kwargs or {}

    # Extract text-slice parameters before passing kwargs to the parser
    text_start = parser_kwargs.pop('_text_start', None)
    text_end   = parser_kwargs.pop('_text_end',   None)

    text = read_text(input_path)
    out_dir = CORPUS / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean OCR noise then write play_text.txt.
    # All downstream work (parsing, byte-span finding) uses the cleaned text so
    # that play_text.txt and quotation_info.csv byte spans stay in sync.
    text = clean_play_text(text)

    # Optionally slice to a sub-section of the file (for anthology volumes)
    if text_start or text_end:
        lines = text.splitlines()
        start_idx = 0
        end_idx = len(lines)
        if text_start:
            for i, ln in enumerate(lines):
                if text_start in ln:
                    start_idx = i
                    break
        if text_end:
            for i, ln in enumerate(lines[start_idx:], start_idx):
                if text_end in ln:
                    end_idx = i
                    break
        text = '\n'.join(lines[start_idx:end_idx])

    play_text_path = out_dir / "play_text.txt"
    play_text_path.write_text(text, encoding='utf-8')
    play_text_bytes = text.encode('utf-8')

    # Parse utterances
    utterances = parser_fn(text, **parser_kwargs)
    if not utterances:
        print(f"  WARNING: no utterances found in {input_path.name}")
        return

    # Build character map
    canon_map, alias_map = build_char_map(utterances)
    speech_counts = Counter(canon_map[raw] for raw, _ in utterances)

    # Write character_info.csv
    chars = assign_ids(alias_map, speech_counts)
    write_character_csv(out_dir / "character_info.csv", chars)

    # Compute addressees (scene co-presence via ±10 utterance window)
    addressees_list = compute_addressees(utterances, canon_map, alias_map, speech_counts)

    # Write quotation_info.csv
    write_quotation_csv(out_dir / "quotation_info.csv", utterances, canon_map,
                        play_text_bytes, addressees_list)

    print(f"  OK  {output_dir_name}: {len(utterances)} quotes, {len(chars)} characters")


# ── PLAY REGISTRY ─────────────────────────────────────────────────────────
# (input_path, output_dir, parser_fn, kwargs)

PLAYS = [
    # ── Beckett ──────────────────────────────────────────────────────────
    (CLEAN_PLAYS/"Beckett_En attendant Godot.txt",
     "Beckett_EnAttendantGodot",       parse_fr_emdash, {}),

    (CLEAN_PLAYS/"Beckett_Endgame.txt",
     "Beckett_Endgame",                parse_en_colon_inline, {}),

    (CLEAN_PLAYS/"Beckett_Fin de partie.txt",
     "Beckett_FinDePartie",            parse_fr_emdash, {}),

    (CLEAN_PLAYS/"Beckett_Waiting for Godot.txt",
     "Beckett_WaitingForGodot",        parse_en_colon_own, {}),

    # ── Becque ───────────────────────────────────────────────────────────
    (CLEAN_PLAYS/"Becque_la parisienne.txt",
     "Becque_LaParisienne",            parse_fr_caps_own, {}),

    (CLEAN_PLAYS/"Becque_The vultures.txt",
     "Becque_TheVultures",             parse_en_caps_own, {}),

    (CLEAN_PLAYS/"Becque_les corbeaux.txt",
     "Becque_LesCorbeaux",             parse_fr_titlecase_emdash, {}),

    (CLEAN_PLAYS/"Becque_the woman of paris.txt",
     "Becque_TheWomanOfParis",         parse_en_caps_own, {}),

    # ── Brieux ───────────────────────────────────────────────────────────
    (CLEAN_PLAYS/"Brieux_Red robe.txt",
     "Brieux_LaRobeRouge_EN",          parse_en_dot_inline, {}),

    (CLEAN_PLAYS/"Brieux_Woman on Her Own.txt",
     "Brieux_WomanOnHerOwn",           parse_en_dot_inline,
     {'_text_end': 'Five Mourners'}),  # anthology vol: stop before Egypt play section

    (CLEAN_OUTPUT/"Brieux_La Femme Seule.txt",
     "Brieux_LaFemmeSeule",            parse_fr_caps_own,
     {'content_start': 'est debout', 'min_count': 2}),  # min_count=2 catches MAUD/ANTOINETTE (appear 2x)

    (CLEAN_OUTPUT/"Brieux_La robe rouge.txt",
     "Brieux_LaRobeRouge",             parse_fr_caps_own, {}),

    # ── Dumas ────────────────────────────────────────────────────────────
    (CLEAN_PLAYS/"Dumas_Camille.txt",
     "Dumas_Camille_EN",               parse_en_dot_inline, {}),

    (CLEAN_PLAYS/"Dumas_Dame aux camelias.txt",
     "Dumas_DameAuxCamelias",          parse_fr_caps_own,
     {'min_count': 5, 'allow_titlecase': False}),

    (CLEAN_PLAYS/"Dumas_Le demi monde.txt",
     "Dumas_LeDemiMonde_FR",           parse_fr_caps_own,
     {'content_start': 'Un salon chez Olivier', 'allow_titlecase': False}),

    (CLEAN_OUTPUT/"Dumas_The Demi monde.txt",
     "Dumas_TheDemiMonde_EN",          parse_en_caps_own,   {'min_count': 3}),

    # ── Genet ────────────────────────────────────────────────────────────
    (CLEAN_PLAYS/"Genet_Balcony.txt",
     "Genet_Balcony_EN",               parse_en_caps_own, {'min_count': 2}),

    (CLEAN_PLAYS/"Genet_The maids.txt",
     "Genet_TheMaids_EN",              parse_en_colon_own, {}),

    (CLEAN_OUTPUT/"Genet_Les bonnes.txt",
     "Genet_LesBonnes_FR",             parse_fr_caps_own,
     {'known_speakers': ['CLAIRE','SOLANGE','MADAME','VOIX DE MADAME']}),

    (CLEAN_OUTPUT/"Genet_Le balcon.txt",
     "Genet_LeBalcon_FR",              parse_fr_caps_own,
     {'min_count': 5, 'content_start': 'PREMIER TABLEAU'}),

    # ── Ionesco ──────────────────────────────────────────────────────────
    (CLEAN_PLAYS/"Ionesco_Bald Soprano.txt",
     "Ionesco_BaldSoprano_EN",         parse_en_colon_inline,
     {'_text_end': 'THE LESSON'}),  # anthology vol: stop before The Lesson

    (CLEAN_PLAYS/"Ionesco_Rhinoceros.txt",
     "Ionesco_Rhinoceros_FR",          parse_fr_caps_own,
     {'content_start': 'ACTE PREMIER'}),  # skip PERSONNAGES section

    (CLEAN_OUTPUT/"Ionesco_Rhinoceros.txt",
     "Ionesco_Rhinoceros_EN",          parse_en_caps_own,
     {'min_count': 2}),

    (CLEAN_OUTPUT/"Ionesco_La cantatrice chauve.txt",
     "Ionesco_LaCantriceChauve_FR",    parse_fr_caps_own,
     {'known_speakers': [
         'MME SMITH','M. SMITH','MME MARTIN','M. MARTIN','MARY','LE POMPIER',
     ]}),
]


if __name__ == "__main__":
    CORPUS.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {CORPUS}/\n")
    for entry in PLAYS:
        input_path, out_name, parser_fn, kwargs = entry
        print(f"Processing {Path(input_path).name} -> {out_name}")
        process_play(input_path, out_name, parser_fn, kwargs)
    print("\nDone.")
