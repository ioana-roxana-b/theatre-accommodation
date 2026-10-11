"""
Is the movement contrast driven by a single author? Absurdist vs realist accommodation with each of
the six authors left out in turn.

  dnm  independent t-test on adjacent dyad-family convergence scores (as in compute_stats.py)
  OR   reply level Mantel-Haenszel odds ratio of B using f after A used f, over play x family strata,
       computed separately for absurdist and realist plays

Reads   outputs/plays_{en,fr}/wordlists/<Play>/analysis_results/convergence_results/conv_by_pair_feature_wordlists.csv
        outputs/alignment/reply_level.csv.gz
Writes  outputs/robustness/leave_one_author_out.csv

Usage:  python scripts/stats/leave_one_author_out.py   (from repo root)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ABSURDIST = {'Beckett', 'Genet', 'Ionesco'}
AUTHORS = ['Beckett', 'Ionesco', 'Genet', 'Becque', 'Brieux', 'Dumas']


def cohend(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / sp


def mh_or(d):
    g = d.assign(n11=(d.a_uses == 1) & (d.b_uses == 1), n10=(d.a_uses == 1) & (d.b_uses == 0),
                 n01=(d.a_uses == 0) & (d.b_uses == 1), n00=(d.a_uses == 0) & (d.b_uses == 0))
    s = g.groupby(['play', 'family'])[['n11', 'n10', 'n01', 'n00']].sum()
    n = s.sum(axis=1)
    return (s.n11 * s.n00 / n).sum() / (s.n10 * s.n01 / n).sum()


def main():
    rows = []
    for lang in ['en', 'fr']:
        for f in Path(f'outputs/plays_{lang}/wordlists').glob(
                '*/analysis_results/convergence_results/conv_by_pair_feature_wordlists.csv'):
            play = f.parts[-4]
            if play.startswith('All_'):
                continue
            d = pd.read_csv(f)
            d['author'] = play.split('_')[0]
            rows.append(d[['conv', 'author']])
    dnm = pd.concat(rows).dropna()
    dnm['absurdist'] = dnm.author.isin(ABSURDIST)

    reply = pd.read_csv('outputs/alignment/reply_level.csv.gz')
    reply = reply[reply.condition == 'adjacent']

    out = []
    for left_out in [None] + AUTHORS:
        d = dnm if left_out is None else dnm[dnm.author != left_out]
        r = reply if left_out is None else reply[reply.author != left_out]
        a, b = d.loc[d.absurdist, 'conv'].values, d.loc[~d.absurdist, 'conv'].values
        _, p = stats.ttest_ind(a, b)
        out.append(dict(left_out=left_out or 'none', dnm_absurdist=a.mean(), dnm_realist=b.mean(),
                        d=cohend(a, b), p=p,
                        or_absurdist=mh_or(r[r.genre == 'Absurdist']), or_realist=mh_or(r[r.genre == 'Realist'])))
    res = pd.DataFrame(out)
    Path('outputs/robustness').mkdir(parents=True, exist_ok=True)
    res.to_csv('outputs/robustness/leave_one_author_out.csv', index=False)
    print(res.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
