import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_alignment_table import WORKS  
from network_preservation import conv_edges, trigger_edges 

OUT = Path('outputs/network_similarity')
DESCRIPTORS = ['recip_corr', 'neg_share', 'out_spread', 'in_spread', 'w_target_deg', 'w_source_deg', 'top_share']


def srho(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return spearmanr(x, y)[0]


def describe(e):
    """Descriptors of one network; e has columns a, b, weight."""
    w = e['weight'].to_numpy()
    sd = np.std(w)
    rev = e.merge(e, left_on=['a', 'b'], right_on=['b', 'a'], suffixes=('', '_rev'))
    rev = rev[rev['a'] < rev['b']]                      # each reciprocal pair once
    deg = pd.concat([e['a'], e['b']]).value_counts()
    absw = np.sort(np.abs(w))[::-1]
    k = max(1, int(np.ceil(len(w) / 4)))
    return {
        'recip_corr': srho(rev['weight'].to_numpy(), rev['weight_rev'].to_numpy()),
        'neg_share': np.mean(w < 0),
        'out_spread': e.groupby('a')['weight'].mean().std() / sd if sd > 0 else np.nan,
        'in_spread': e.groupby('b')['weight'].mean().std() / sd if sd > 0 else np.nan,
        'w_target_deg': srho(w, e['b'].map(deg).to_numpy()),
        'w_source_deg': srho(w, e['a'].map(deg).to_numpy()),
        'top_share': absw[:k].sum() / absw.sum() if absw.sum() > 0 else np.nan,
    }


def distance_matrix(desc_en, desc_fr):
    """RMS distance between z-scored descriptor vectors (z over all 24 networks)."""
    both = pd.concat([desc_en, desc_fr])
    z = (both - both.mean()) / both.std()
    ze, zf = z.iloc[:len(desc_en)].to_numpy(), z.iloc[len(desc_en):].to_numpy()
    diff = ze[:, None, :] - zf[None, :, :]
    return np.sqrt(np.nanmean(diff ** 2, axis=2))


def stats_of(D):
    n = len(D)
    diag = np.diag(D)
    off = D[~np.eye(n, dtype=bool)]
    identified = int(np.sum(np.argmin(D, axis=1) == np.arange(n)))
    return off.mean() - diag.mean(), identified


def analyse(weight_name, get_edges, rng, n_perm):
    works = list(WORKS)
    nets_en = [get_edges(en, 'en') for en, _ in WORKS.values()]
    nets_fr = [get_edges(fr, 'fr') for _, fr in WORKS.values()]
    desc_en = pd.DataFrame([describe(e) for e in nets_en], index=works)
    desc_fr = pd.DataFrame([describe(e) for e in nets_fr], index=works)
    D = distance_matrix(desc_en, desc_fr)
    adv, ident = stats_of(D)

    # matching null: shuffle which FR network belongs to which work
    null_match = []
    for _ in range(n_perm):
        p = rng.permutation(len(works))
        null_match.append(stats_of(D[:, p]))
    null_match = np.array(null_match)

    # topology null: shuffle weights within every network, recompute everything
    def shuffled(e):
        return e.assign(weight=rng.permutation(e['weight'].to_numpy()))
    null_topo = []
    for _ in range(n_perm):
        de = pd.DataFrame([describe(shuffled(e)) for e in nets_en], index=works)
        df_ = pd.DataFrame([describe(shuffled(e)) for e in nets_fr], index=works)
        null_topo.append(stats_of(distance_matrix(de, df_)))
    null_topo = np.array(null_topo)

    def p_ge(obs, null):
        return (1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1)

    summary = pd.DataFrame([
        dict(statistic='diag_advantage', observed=adv,
             matching_null_mean=null_match[:, 0].mean(), matching_p=p_ge(adv, null_match[:, 0]),
             topology_null_mean=null_topo[:, 0].mean(), topology_p=p_ge(adv, null_topo[:, 0])),
        dict(statistic='identified (of 12)', observed=ident,
             matching_null_mean=null_match[:, 1].mean(), matching_p=p_ge(ident, null_match[:, 1]),
             topology_null_mean=null_topo[:, 1].mean(), topology_p=p_ge(ident, null_topo[:, 1])),
    ])
    rank = pd.DataFrame({'work': works,
                         'own_distance': np.diag(D),
                         'rank_of_own_translation': [int((D[i] < D[i, i]).sum()) + 1 for i in range(len(works))]})
    desc = pd.concat([desc_en.assign(language='English'), desc_fr.assign(language='French')]).rename_axis('work')
    return desc.reset_index(), pd.DataFrame(D, index=works, columns=works), summary, rank


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n_perm', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    reply = pd.read_csv('outputs/alignment/reply_level.csv.gz')
    weights = {'trigger': lambda play, lang: trigger_edges(reply, play), 'conv': conv_edges}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 200)
    for name, fn in weights.items():
        rng = np.random.default_rng(args.seed)
        desc, D, summary, rank = analyse(name, fn, rng, args.n_perm)
        desc.to_csv(OUT / f'descriptors_{name}.csv', index=False)
        D.to_csv(OUT / f'distance_matrix_{name}.csv')
        summary.to_csv(OUT / f'summary_{name}.csv', index=False)
        rank.to_csv(OUT / f'own_translation_rank_{name}.csv', index=False)
        print(f'\n=== edge weight: {name} ===')
        print(summary.round(4).to_string(index=False))
        print(rank.round(3).to_string(index=False))
    print(f'\nOutputs in {OUT}/')


if __name__ == '__main__':
    main()
