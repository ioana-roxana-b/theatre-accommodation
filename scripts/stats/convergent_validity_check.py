"""
Convergent validity check: univariate comparisons vs the mixed effects model.

For genre, language and gender dyad, prints the simple two-group comparison on the
adjacent convergence scores (Welch t-test, Cohen's d) next to the likelihood-ratio
test for the same factor from scripts/stats/mixed_effects_model.py (run that first).

Usage:  python scripts/stats/convergent_validity_check.py   (from repo root)
"""
import os
import numpy as np
import pandas as pd
import scipy.stats as stats

BASES = {'English': 'outputs/plays_en/wordlists', 'French': 'outputs/plays_fr/wordlists'}
ABSURDIST = {'Beckett', 'Genet', 'Ionesco'}


def load():
    dfs = []
    for language, base in BASES.items():
        for play in sorted(os.listdir(base)):
            path = f'{base}/{play}/analysis_results/convergence_results/conv_by_pair_feature_wordlists.csv'
            if play == 'All_Plays' or not os.path.exists(path):
                continue
            df = pd.read_csv(path, encoding='utf-8-sig')
            df['play'] = play
            df['language'] = language
            df['genre'] = 'Absurdist' if play.split('_')[0] in ABSURDIST else 'Realist'
            df['gender_dyad'] = ['-'.join(sorted([str(a), str(b)])) if pd.notna(a) and pd.notna(b) else 'unknown'
                                 for a, b in zip(df['a_gender'], df['b_gender'])]
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data[data['conv'].notna()]


def cohend(a, b):
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / pooled


data = load()
lrt = pd.read_csv('outputs/mixed_effects_lrt_importance.csv').set_index('factor')
print(f'{len(data)} dyad x family observations from {data["play"].nunique()} plays\n')

comparisons = [
    ('genre', 'Absurdist', 'Realist'),
    ('language', 'English', 'French'),
    ('gender_dyad', 'F-F', 'M-M'),
]
print(f'{"factor":12s} {"comparison":22s} {"means":17s} {"d":>6s} {"p (t-test)":>11s}   {"model LRT":s}')
for factor, g1, g2 in comparisons:
    a = data.loc[data[factor] == g1, 'conv'].to_numpy()
    b = data.loc[data[factor] == g2, 'conv'].to_numpy()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    r = lrt.loc[factor]
    print(f'{factor:12s} {g1 + " vs " + g2:22s} {np.mean(a):.4f} / {np.mean(b):.4f}  {cohend(a, b):+.3f} {p:11.4g}   '
          f'chi2({int(r["df"])})={r["chi2"]:.2f}, p={r["p"]:.4g}')

print("""
Univariate tests treat every dyad x family row as independent. The model accounts for
play and dyad clustering and for the other factors, so a factor that is significant only
univariately differs between plays but cannot be separated from play-level variation
with 24 plays.""")
