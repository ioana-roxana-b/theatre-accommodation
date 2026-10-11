from pathlib import Path
import pandas as pd


WORKS = {
    'Godot':      ('Beckett_WaitingForGodot',  'Beckett_EnAttendantGodot'),
    'Endgame':    ('Beckett_Endgame',          'Beckett_FinDePartie'),
    'Corbeaux':   ('Becque_TheVultures',       'Becque_LesCorbeaux'),
    'Parisienne': ('Becque_TheWomanOfParis',   'Becque_LaParisienne'),
    'RobeRouge':  ('Brieux_LaRobeRouge_EN',    'Brieux_LaRobeRouge'),
    'FemmeSeule': ('Brieux_WomanOnHerOwn',     'Brieux_LaFemmeSeule'),
    'Camelias':   ('Dumas_Camille_EN',         'Dumas_DameAuxCamelias'),
    'DemiMonde':  ('Dumas_TheDemiMonde_EN',    'Dumas_LeDemiMonde_FR'),
    'Balcon':     ('Genet_Balcony_EN',         'Genet_LeBalcon_FR'),
    'Bonnes':     ('Genet_TheMaids_EN',        'Genet_LesBonnes_FR'),
    'Cantatrice': ('Ionesco_BaldSoprano_EN',   'Ionesco_LaCantriceChauve_FR'),
    'Rhinoceros': ('Ionesco_Rhinoceros_EN',    'Ionesco_Rhinoceros_FR'),
}
ABSURDIST = {'Beckett', 'Genet', 'Ionesco'}

# column prefix in pairs_features.csv -> family name used in the paper / other scripts
FAMILIES = {
    'article': 'articles', 'auxverb': 'auxiliary_verbs', 'conj': 'conjunctions',
    'adverb': 'adverbs', 'ppron': 'personal_pronouns', 'ipron': 'impersonal_pronouns',
    'negate': 'negations', 'prep': 'prepositions', 'quant': 'quantifiers',
}

CONDITIONS = {'adjacent': '', 'nonadjacent': '_nonadjacent', 'randomized': '_randomized'}

OUT = Path('outputs/alignment/reply_level.csv.gz')


def gender_dyad(a, b):
    if pd.isna(a) or pd.isna(b):
        return 'unknown'
    return '-'.join(sorted([str(a), str(b)]))


def load_play(play, work, language, condition):
    lang_dir = 'plays_en' if language == 'English' else 'plays_fr'
    path = Path(f'outputs/{lang_dir}/wordlists/{play}/pairs_features{CONDITIONS[condition]}.csv')
    df = pd.read_csv(path)
    author = play.split('_')[0]
    base = pd.DataFrame({
        'condition': condition,
        'pair_id': range(len(df)),
        'play': play,
        'work': work,
        'language': language,
        'author': author,
        'genre': 'Absurdist' if author in ABSURDIST else 'Realist',
        'dyad': play + '::' + df['A'].astype(str) + '->' + df['B'].astype(str),
        'b_speaker': play + '::' + df['B'].astype(str),
        'gender_dyad': [gender_dyad(a, b) for a, b in zip(df['A_gender'], df['B_gender'])],
        'importance_dyad': df['A_category'].astype(str) + '->' + df['B_category'].astype(str),
        'a_len': df['i_tokens'],
        'b_len': df['j_tokens'],
    })
    rows = []
    for prefix, family in FAMILIES.items():
        part = base.copy()
        part['family'] = family
        part['a_uses'] = (df[f'{prefix}_i_present'] > 0).astype(int)
        part['b_uses'] = (df[f'{prefix}_j_present'] > 0).astype(int)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def main():
    parts = []
    for work, (en, fr) in WORKS.items():
        for condition in CONDITIONS:
            parts.append(load_play(en, work, 'English', condition))
            parts.append(load_play(fr, work, 'French', condition))
    data = pd.concat(parts, ignore_index=True)
    # turns with no tokens can't contain a family; they carry no information
    data = data[(data['a_len'] > 0) & (data['b_len'] > 0)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index=False)
    print(f'Saved {OUT}: {len(data):,} rows, {data["play"].nunique()} plays, {data["work"].nunique()} works')
    pairs = data.drop_duplicates(['condition', 'play', 'pair_id'])
    print(pairs.groupby('condition').agg(pairs=('pair_id', 'size'), dyads=('dyad', 'nunique')).to_string())
    print(data[data['condition'] == 'adjacent'].groupby('family')[['a_uses', 'b_uses']].mean().round(3).to_string())


if __name__ == '__main__':
    main()
