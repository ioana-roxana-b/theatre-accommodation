"""
fix_character_genders.py
Patches Gender column in every data/plays_dialogue/*/character_info.csv.
Leaves Category (importance) untouched — that is set by quote frequency.
"""

import csv, re
from pathlib import Path

CORPUS = Path("data/plays_dialogue")

# ── Name → gender lookup (normalised to uppercase, no accents) ───────────────
GENDER_MAP = {
    # ── Beckett ──────────────────────────────────────────────────────────────
    'BOY': 'M', 'GARCON': 'M',
    'ESTRAGON': 'M', 'ESTRAPON': 'M',
    'LUCKY': 'M', 'POZZO': 'M',
    'VLADIMIR': 'M', 'VL ADIMIR': 'M',
    'CLOV': 'M', 'HAMM': 'M', 'NAGG': 'M', 'NELL': 'F',

    # ── Becque ───────────────────────────────────────────────────────────────
    'ADELE': 'F', 'ADELE': 'F',
    'CLOTILDE': 'F',
    'DU MESNIL': 'M', 'LAFONT': 'M', 'SIMPSON': 'M',
    'AUGUSTE': 'M', 'BLANCHE': 'F',
    'BOURDON': 'M', 'DUPUIS': 'M',
    'JUDITH': 'F', 'LEFORT': 'M', 'MARIE': 'F',
    'MERCKENS': 'M', 'ROSALIE': 'F',
    'TEISSIER': 'M', 'VIGNERON': 'M',
    'THE DOCTOR': 'M', 'LE MEDECIN': 'M',
    'MRS': 'F',
    'BLANCHE ETJUDITH': 'F',
    'LES ENFANTS': 'U',

    # ── Brieux ───────────────────────────────────────────────────────────────
    'ANTOINETTE': 'F', 'CAROLINE LEGRAND': 'F',
    'LA MERE BOUGNE': 'F', 'MOTHER BOUGNE': 'F',
    'LE DELEGUE': 'M', 'DELEGATE': 'M',
    'LE GARCON': 'M', 'LE GARCON DE BUREAU': 'M',
    'LUCIENNE': 'F',
    'MADEMOISELLE DE MEURIOT': 'F', 'MADEMOISELLE GREGOIRE': 'F',
    'MAUD': 'F', 'RENE': 'M', 'THERESE': 'F', 'VINCENT': 'M',
    'BERTHA': 'F', 'BERTHE': 'F',
    'BUNERAT': 'M', 'CATIALEENA': 'F', 'CATIALENA': 'F',
    'ETCHEPARE': 'M',
    'LOFFICIER': 'M', 'OFFICER': 'M',
    'LA BOUZULE': 'F', 'LA BOUZOLE': 'F',
    'LE CONCIERGE': 'M', 'DOORKEEPER': 'M',
    'LE DOMESTIQUE': 'M', 'THE MANSERVANT': 'M',
    'LE GREFFIER': 'M', 'RECORDER': 'M', 'THE RECORDER': 'M',
    'LE PROCUREUR GENERAL': 'M', 'ATTORNEY-GENERAL': 'M',
    'LE PRESIDENT DES ASSISES': 'M', 'THE PRESIDENT': 'M',
    'MONDOUBLEAU': 'M', 'MOUZON': 'M', 'VAGRET': 'M',
    'YANETTA': 'F', 'ARDEUIL': 'M', 'BRIDET': 'M',
    'DELORME': 'M', 'OLD MADAME ETCHEPARE': 'F', 'PLACAT': 'M',
    'THE MOTHER': 'F',
    'CHARPIN': 'M', 'CONSTANCE': 'F', 'DESCHAUME': 'M',
    'FELIAT': 'M', 'GIRARD': 'M', 'GUERET': 'M',
    'NERISSE': 'M', 'WOMEN': 'F',

    # ── Dumas ────────────────────────────────────────────────────────────────
    'ARMAND': 'M', 'CAMILLE': 'F', 'DUVAL': 'M',
    'GUSTAVE': 'M', 'GASTON': 'M',
    'NANINE': 'F', 'NICHETTE': 'F',
    'OLIMPE': 'F', 'OLYMPE': 'F',
    'PRUDENCE': 'F', 'VARVILLE': 'M',
    'ANAIS': 'F', 'ARTHUR': 'M', 'LE COMTE': 'M',
    'MARGUERITE': 'F', 'SAINT-GAUDENS': 'M',
    'HIPPOLYTE': 'M', 'LE MARQUIS': 'M', 'MARQUIS': 'M',
    'MARCELLE': 'F', 'MARCELLA': 'F',
    'OLIVIER': 'M', 'OLIVER': 'M',
    'RAYMOND': 'M', 'SOPHIE': 'F',
    'SUZANNE': 'F', 'VALENTINE': 'F', 'VALENTINA': 'F', 'SUSAN': 'F',
    'COUNTESS': 'F', 'SERVANT': 'M',
    'MESSENGER': 'M',

    # ── Genet ────────────────────────────────────────────────────────────────
    'CARMEN': 'F', 'CHANTAL': 'F', 'IRMA': 'F',
    'ROGER': 'M', 'ARTHUR': 'M',
    'THE BISHOP': 'M', 'LEVEQUE': 'M',
    'THE CHIEF OF POLICE': 'M', 'LE CHEF DE LA POLICE': 'M',
    'THE ENVOY': 'M', 'LENVOYE': 'M',
    'THE EXECUTIONER': 'M', 'LE BOURREAU': 'M',
    'THE GENERAL': 'M', 'LE GENERAL': 'M',
    'THE GIRL': 'F', 'LA FILLE': 'F',
    'THE JUDGE': 'M', 'LE JUGE': 'M',
    'THE QUEEN': 'F', 'LA REINE': 'F',
    'THE THIEF': 'F', 'LA VOLEUSE': 'F',
    'LA FEMME': 'F', 'LHOMME': 'M',
    'LE 1ER PHOTOGRAPHE': 'M', 'LE 2E PHOTOGRAPHE': 'M', 'LE 3E PHOTOGRAPHE': 'M',
    'LESCLAVE': 'U', 'THE SLAVE': 'U',
    'CLAIRE': 'F', 'SOLANGE': 'F', 'VOIX DE MADAME': 'F',
    'THE WOMAN': 'F',

    # ── Ionesco ──────────────────────────────────────────────────────────────
    'FIRE CHIEF': 'M', 'LE POMPIER': 'M',
    'MARY': 'F',
    'BERENGER': 'M', 'BERENGER': 'M',
    'BOTARD': 'M', 'DAISY': 'F', 'DUDARD': 'M',
    'FIREMAN': 'M', 'JEAN': 'M',
    'PAPILLON': 'M',
    'THE CAFE PROPRIETOR': 'M', 'LE PATRON': 'M',
    'THE GROCER': 'M', 'LEPICIER': 'M',
    'THE LITTLE OLD MAN': 'M', 'LE PETIT VIEUX': 'M',
    'THE LOGICIAN': 'M', 'LE LOGICIEN': 'M',
    'THE OLD GENTLEMAN': 'M', 'LE VIEUX MONSIEUR': 'M',
    'THE WAITRESS': 'F', 'LA SERVEUSE': 'F',
    'THE HOUSEWIFE': 'F', 'LA MENAGERE': 'F',
}

# ── Title-prefix rules (applied after GENDER_MAP lookup fails) ───────────────
_FEMALE_PREFIXES = re.compile(
    r'^(MME|MADAME|MRS\.?|MISS|MADEMOISELLE|LA |LES DAMES|DAME|SIGNORA)\b',
    re.IGNORECASE,
)
_MALE_PREFIXES = re.compile(
    r'^(M\.|MR\.?|MONSIEUR|MONSIUER|LE |BARON|LORD|SIR |COUNT |DUKE |DOCTOR |DR\.?|COLONEL|CAPTAIN)\b',
    re.IGNORECASE,
)


def _strip_accents(s: str) -> str:
    import unicodedata
    nfkd = unicodedata.normalize('NFD', s)
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')


def infer_gender(name: str) -> str | None:
    """Return 'M', 'F', or None if not determinable."""
    key = _strip_accents(name.strip().upper())
    key_nospace = re.sub(r'[^A-Z0-9 ]', '', key).strip()

    if key_nospace in GENDER_MAP:
        return GENDER_MAP[key_nospace]

    # Partial match for compound names (e.g. "BLANCHE ETJUDITH" → 'F')
    for k, g in GENDER_MAP.items():
        if k and k in key_nospace:
            return g

    if _FEMALE_PREFIXES.match(name.strip()):
        return 'F'
    if _MALE_PREFIXES.match(name.strip()):
        return 'M'

    return None


def fix_file(path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig')))
    if not rows:
        return

    changed = 0
    for row in rows:
        if row.get('Gender', 'U') != 'U':
            continue
        inferred = infer_gender(row['Main Name'])
        if inferred and inferred != 'U':
            row['Gender'] = inferred
            changed += 1

    if changed:
        fieldnames = list(rows[0].keys())
        with path.open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  {path.parent.name}: {changed} gender(s) updated")
    else:
        print(f"  {path.parent.name}: no changes")


if __name__ == '__main__':
    files = sorted(CORPUS.glob('*/character_info.csv'))
    print(f"Fixing genders in {len(files)} character_info files...\n")
    for f in files:
        fix_file(f)
    print("\nDone.")
