import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_alignment_table import WORKS  

OUT = Path('outputs/network_preservation')
ALIGNMENT = Path(__file__).resolve().parent / 'character_alignment.csv'
MIN_GROUP = 3      # trigger weight: pairs needed with and without A using the family
MIN_FAMILIES = 3   # trigger weight: families needed per edge
MIN_EDGES = 4      # matched edges needed to compute a per-work rho
MIN_NODES = 3      # aligned characters needed for node-role rho


# ── Edge weights ───────────────────────────────────────────────────────────
def conv_edges(play, lang):
    path = (f'outputs/plays_{lang}/wordlists/{play}/analysis_results/convergence_results/'
            'conv_edge_summary_wordlists.csv')
    e = pd.read_csv(path, encoding='utf-8-sig')
    return e.rename(columns={'a_speaker': 'a', 'b_speaker': 'b', 'mean_conv': 'weight'})[['a', 'b', 'weight', 'n_pairs']]


def trigger_edges(reply, play):
    d = reply[(reply['play'] == play) & (reply['condition'] == 'adjacent')]
    g = d.groupby(['dyad', 'family', 'a_uses'])['b_uses'].agg(['mean', 'size']).unstack('a_uses')
    ok = (g[('size', 0)] >= MIN_GROUP) & (g[('size', 1)] >= MIN_GROUP)
    eff = (g[('mean', 1)] - g[('mean', 0)])[ok].rename('effect').reset_index()
    e = eff.groupby('dyad')['effect'].agg(weight='mean', n_families='size').reset_index()
    e = e[e['n_families'] >= MIN_FAMILIES]
    n_pairs = d.drop_duplicates('pair_id').groupby('dyad').size().rename('n_pairs')
    e = e.join(n_pairs, on='dyad')
    names = e['dyad'].str.split('::', n=1).str[1].str.split('->', n=1)
    e['a'], e['b'] = names.str[0], names.str[1]
    return e[['a', 'b', 'weight', 'n_pairs']]


# ── Statistics ─────────────────────────────────────────────────────────────
def rho(x, y):
    return spearmanr(x, y)[0] if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan


def partial_rho(x, y, nx, ny):
    """Spearman rho after regressing each version's weight ranks on its log(n_pairs) ranks.
    Rules out that EN and FR agree only because both follow dyad size."""
    def resid(v, n):
        r, ln = pd.Series(v).rank().to_numpy(), pd.Series(np.log(n)).rank().to_numpy()
        return r - np.polyval(np.polyfit(ln, r, 1), ln)
    return np.corrcoef(resid(x, nx), resid(y, ny))[0, 1]


def sign_agreement(x, y):
    return np.mean(np.sign(x) == np.sign(y))


def expected_sign_agreement(x, y):
    px, py = np.mean(x > 0), np.mean(y > 0)
    return px * py + (1 - px) * (1 - py)


def perm_p(x, y, stat, rng, n_perm):
    obs = stat(x, y)
    null = np.array([stat(x, rng.permutation(y)) for _ in range(n_perm)])
    return (1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1)


def strengths(edges):
    out_s = edges.groupby('a')['weight'].mean().rename('out_strength')
    in_s = edges.groupby('b')['weight'].mean().rename('in_strength')
    return pd.concat([out_s, in_s], axis=1)


# ── Main ───────────────────────────────────────────────────────────────────
def analyse(weight_name, get_edges, alignment, rng, n_perm, min_pairs):
    matched_all, per_work, roles_all, perm_inputs = [], [], [], []
    for work, (en_play, fr_play) in WORKS.items():
        E, F = get_edges(en_play, 'en'), get_edges(fr_play, 'fr')
        E, F = E[E['n_pairs'] >= min_pairs], F[F['n_pairs'] >= min_pairs]
        amap = alignment[alignment['work'] == work].set_index('en')['fr'].to_dict()

        known_en, known_fr = set(E['a']) | set(E['b']), set(F['a']) | set(F['b'])
        for en_name, fr_name in amap.items():
            if en_name not in known_en and fr_name not in known_fr:
                continue  # character without edges in either version (e.g. filtered out)
            if en_name not in known_en or fr_name not in known_fr:
                missing = en_name if en_name not in known_en else fr_name
                print(f'  [{weight_name}] {work}: aligned name without edges in one version: {missing!r}')

        E = E.assign(a_fr=E['a'].map(amap), b_fr=E['b'].map(amap)).dropna(subset=['a_fr', 'b_fr'])
        m = E.merge(F, left_on=['a_fr', 'b_fr'], right_on=['a', 'b'], suffixes=('_en', '_fr'))
        m = m.rename(columns={'a_en': 'a', 'b_en': 'b'})[['a', 'b', 'a_fr', 'b_fr', 'weight_en', 'weight_fr',
                                                          'n_pairs_en', 'n_pairs_fr']]
        m.insert(0, 'work', work)
        matched_all.append(m)
        x, y = m['weight_en'].to_numpy(), m['weight_fr'].to_numpy()

        row = dict(work=work, edges_en=len(E), edges_fr=len(F), matched_edges=len(m))
        if len(m) >= MIN_EDGES:
            row.update(rho=rho(x, y), rho_p=perm_p(x, y, rho, rng, n_perm),
                       rho_partial_npairs=partial_rho(x, y, m['n_pairs_en'].to_numpy(), m['n_pairs_fr'].to_numpy()),
                       sign_agree=sign_agreement(x, y), sign_expected=expected_sign_agreement(x, y),
                       sign_p=perm_p(x, y, sign_agreement, rng, n_perm))
            perm_inputs.append((x, y))

        # node roles from all edges of each version, compared over aligned characters
        se, sf = strengths(E.drop(columns=['a_fr', 'b_fr'])), strengths(F)
        se.index = se.index.map(amap)
        r = se.join(sf, lsuffix='_en', rsuffix='_fr', how='inner')
        r.insert(0, 'work', work)
        roles_all.append(r.rename_axis('character_fr').reset_index())
        for role in ['out_strength', 'in_strength']:
            rr = r[[f'{role}_en', f'{role}_fr']].dropna()
            if len(rr) >= MIN_NODES:
                a_, b_ = rr.iloc[:, 0].to_numpy(), rr.iloc[:, 1].to_numpy()
                row[f'{role}_rho'] = rho(a_, b_)
                row[f'{role}_p'] = perm_p(a_, b_, rho, rng, n_perm)
                row[f'{role}_n'] = len(rr)
        per_work.append(row)

    matched = pd.concat(matched_all, ignore_index=True)
    per_work = pd.DataFrame(per_work)
    roles = pd.concat(roles_all, ignore_index=True)

    # pooled: mean per-work rho and overall sign agreement, shuffling within every work at once
    obs_rho = np.nanmean([rho(x, y) for x, y in perm_inputs])
    obs_sign = np.mean(np.concatenate([np.sign(x) == np.sign(y) for x, y in perm_inputs]))
    null_rho, null_sign = [], []
    for _ in range(n_perm):
        shuffled = [(x, rng.permutation(y)) for x, y in perm_inputs]
        null_rho.append(np.nanmean([rho(x, y) for x, y in shuffled]))
        null_sign.append(np.mean(np.concatenate([np.sign(x) == np.sign(y) for x, y in shuffled])))
    null_rho, null_sign = np.array(null_rho), np.array(null_sign)
    pooled = pd.DataFrame([
        dict(measure='mean edge rho', observed=obs_rho, null_mean=null_rho.mean(),
             partial_npairs=per_work['rho_partial_npairs'].mean(),
             p=(1 + np.sum(null_rho >= obs_rho - 1e-12)) / (n_perm + 1), works=len(perm_inputs),
             edges=sum(len(x) for x, _ in perm_inputs)),
        dict(measure='sign agreement', observed=obs_sign, null_mean=null_sign.mean(),
             p=(1 + np.sum(null_sign >= obs_sign - 1e-12)) / (n_perm + 1), works=len(perm_inputs),
             edges=sum(len(x) for x, _ in perm_inputs)),
    ])
    return matched, per_work, roles, pooled


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--n_perm', type=int, default=10000)
    ap.add_argument('--min_pairs', type=int, default=0,
                    help='only compare edges with at least this many A->B pairs in both versions')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    alignment = pd.read_csv(ALIGNMENT, encoding='utf-8')
    reply = pd.read_csv('outputs/alignment/reply_level.csv.gz')
    weights = {
        'conv': conv_edges,
        'trigger': lambda play, lang: trigger_edges(reply, play),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f'_min{args.min_pairs}' if args.min_pairs else ''
    pd.set_option('display.width', 200)
    for name, fn in weights.items():
        rng = np.random.default_rng(args.seed)
        matched, per_work, roles, pooled = analyse(name, fn, alignment, rng, args.n_perm, args.min_pairs)
        for label, df in [('matched_edges', matched), ('per_work', per_work),
                          ('node_roles', roles), ('pooled', pooled)]:
            df.to_csv(OUT / f'{label}_{name}{suffix}.csv', index=False)
        print(f'\n=== edge weight: {name} ===')
        print(per_work.round(3).to_string(index=False))
        print(pooled.round(4).to_string(index=False))
    print(f'\nOutputs in {OUT}/')


if __name__ == '__main__':
    main()
