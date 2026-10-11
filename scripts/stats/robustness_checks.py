import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocessing.features_wordlists import _tokenize  # noqa: E402
from preprocessing.function_word_lists import LISTS  # noqa: E402
from scripts.stats.build_alignment_table import WORKS, FAMILIES, ABSURDIST  # noqa: E402

OUT = Path('outputs/robustness')
MIN_SUPPORT = 3
MIN_RESP = None  
PREFIXES = list(FAMILIES)                       
FR_AMBIGUOUS_ARTICLES = {'le', 'la', 'les', 'l'}


# ── Data ───────────────────────────────────────────────────────────────────
def plays():
    for en, fr in WORKS.values():
        yield en, 'en'
        yield fr, 'fr'


def load_pairs(play, lang, suffix=''):
    d = pd.read_csv(f'outputs/plays_{lang}/wordlists/{play}/pairs_features{suffix}.csv')
    if 'i_addressees' not in d.columns:
        meta = pd.read_csv(f'outputs/plays_{lang}/wordlists/{play}/pairs_A_to_B{suffix}.csv')
        assert len(meta) == len(d)
        d['i_addressees'] = meta['i_addressees'].values
    d['play'], d['lang'] = play, lang.upper()
    d['author'] = play.split('_')[0]
    d['genre'] = np.where(d['author'].isin(ABSURDIST), 'Absurdist', 'Realist')
    return d


def n_addressees(s):
    """Candidate addressees of A's turn, stored as 'X; Y; Z'."""
    s = str(s).strip()
    if s.startswith('['):
        return len(ast.literal_eval(s))
    return len([x for x in s.split(';') if x.strip()])


# ── Convergence ────────────────────────────────────────────────────────────
def convergence(pairs, min_resp=MIN_RESP):
    """One row per (play, A, B, family) passing the thresholds, with conv = p1 - p0."""
    keys = ['play', 'A', 'B']
    meta = pairs.groupby(keys, sort=False)[['lang', 'author', 'genre']].first()
    out = []
    for f in PREFIXES:
        i = pairs[f'{f}_i_present'].to_numpy()
        j = pairs[f'{f}_j_present'].to_numpy()
        s = (pd.DataFrame({'n': 1, 'i': i, 'j': j, 'ij': i * j}, index=pairs.index)
             .join(pairs[keys]).groupby(keys, sort=False).sum())
        keep = s['i'] >= MIN_SUPPORT
        if min_resp:
            keep &= s['ij'] >= min_resp
        s = s[keep]
        out.append(pd.DataFrame({'family': FAMILIES[f], 'conv': s['ij'] / s['i'] - s['j'] / s['n']}, index=s.index))
    res = pd.concat(out).join(meta).reset_index()
    return res[['play', 'lang', 'author', 'genre', 'A', 'B', 'family', 'conv']]


def _dyad_index(pairs):
    return list(pairs.groupby(['play', 'A', 'B'], sort=False).indices.values())


def shuffle_replies(pairs, seed, groups=None):
    """Randomized control: B's replies shuffled within each dyad (A's turns stay in place)."""
    rng = np.random.default_rng(seed)
    groups = groups if groups is not None else _dyad_index(pairs)
    perm = np.arange(len(pairs))
    for idx in groups:
        perm[idx] = rng.permutation(idx)
    out = pairs.copy()
    jcols = [f'{f}_j_present' for f in PREFIXES]
    out[jcols] = pairs[jcols].to_numpy()[perm]
    return out


def cohend(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(a) - np.mean(b)) / sp


def compare(a, b):
    t, p = stats.ttest_ind(a, b)
    return dict(mean_1=np.mean(a), mean_2=np.mean(b), n_1=len(a), n_2=len(b), d=cohend(a, b), t=t, p=p)


# ── Checks ─────────────────────────────────────────────────────────────────
def check_seeds(adj_pairs, n_seeds):
    adj = convergence(adj_pairs)
    rows = []
    for seed in range(n_seeds):
        rnd = convergence(shuffle_replies(adj_pairs, seed))
        r = compare(adj['conv'], rnd['conv'])
        abs_r = compare(adj.loc[adj.genre == 'Absurdist', 'conv'], rnd.loc[rnd.genre == 'Absurdist', 'conv'])
        rea_r = compare(adj.loc[adj.genre == 'Realist', 'conv'], rnd.loc[rnd.genre == 'Realist', 'conv'])
        rows.append(dict(seed=seed, adj_mean=r['mean_1'], rand_mean=r['mean_2'], d=r['d'], p=r['p'],
                         d_absurdist=abs_r['d'], d_realist=rea_r['d']))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'seeds.csv', index=False)
    print(f'\n== Randomized baseline over {n_seeds} seeds (adjacent mean {adj.conv.mean():.4f}, '
          f'n = {len(adj)} dyad-family scores)')
    print(res[['rand_mean', 'd', 'p', 'd_absurdist', 'd_realist']].describe(percentiles=[.025, .975])
          .loc[['mean', 'std', 'min', '2.5%', '97.5%', 'max']].round(4).to_string())


def check_addressee(adj_pairs, non_pairs):
    single = lambda d: d[d['i_addressees'].map(n_addressees) == 1]
    a_p, n_p = single(adj_pairs), single(non_pairs)
    adj, non, rnd = convergence(a_p), convergence(n_p), convergence(shuffle_replies(a_p, 123))
    share = len(a_p) / len(adj_pairs)
    rows = [
        dict(test='adjacent vs randomized', **compare(adj.conv, rnd.conv)),
        dict(test='adjacent vs non-adjacent', **compare(adj.conv, non.conv)),
        dict(test='absurdist vs realist (adjacent)',
             **compare(adj.loc[adj.genre == 'Absurdist', 'conv'], adj.loc[adj.genre == 'Realist', 'conv'])),
        dict(test='English vs French (adjacent)',
             **compare(adj.loc[adj.lang == 'EN', 'conv'], adj.loc[adj.lang == 'FR', 'conv'])),
    ]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'single_addressee.csv', index=False)
    print(f'\n== Single candidate addressee: {len(a_p)} of {len(adj_pairs)} adjacent pairs ({100 * share:.1f}%)')
    print(res.round(4).to_string(index=False))


def check_french(adj_pairs):
    articles_fr = set(LISTS['fr']['article']) - FR_AMBIGUOUS_ARTICLES
    mod = adj_pairs.copy()
    fr = mod['lang'] == 'FR'
    for side in ['i', 'j']:
        toks = mod.loc[fr, f'{side}_text'].astype(str).map(_tokenize)
        mod.loc[fr, f'article_{side}_present'] = toks.map(lambda t: int(any(w in articles_fr for w in t))).values
    base, alt = convergence(adj_pairs), convergence(mod)
    rows = []
    for name, c in [('original', base), ('without le/la/les/l\'', alt)]:
        rows.append(dict(version=name, test='absurdist vs realist',
                         **compare(c.loc[c.genre == 'Absurdist', 'conv'], c.loc[c.genre == 'Realist', 'conv'])))
        rows.append(dict(version=name, test='English vs French',
                         **compare(c.loc[c.lang == 'EN', 'conv'], c.loc[c.lang == 'FR', 'conv'])))
        art = c[c.family == 'articles']
        rows.append(dict(version=name, test='articles: absurdist vs realist',
                         **compare(art.loc[art.genre == 'Absurdist', 'conv'], art.loc[art.genre == 'Realist', 'conv'])))
        frc = c[c.lang == 'FR']
        rows.append(dict(version=name, test='French only: absurdist vs realist',
                         **compare(frc.loc[frc.genre == 'Absurdist', 'conv'], frc.loc[frc.genre == 'Realist', 'conv'])))
    res = pd.DataFrame(rows)

    # cross-language edge preservation with the modified French edge weights
    m = pd.read_csv('outputs/network_preservation/matched_edges_conv.csv', header=0)
    m.columns = ['work', 'a', 'b', 'a_fr', 'a_fr_dup', 'b_fr', 'b_fr_dup', 'weight_en', 'weight_fr', 'n_en', 'n_fr']
    fr_play = {w: f for w, (_, f) in WORKS.items()}
    w_alt = alt[alt.lang == 'FR'].groupby(['play', 'A', 'B'])['conv'].mean().rename('weight_fr_alt').reset_index()
    m['play'] = m['work'].map(fr_play)
    m = m.merge(w_alt, left_on=['play', 'a_fr', 'b_fr'], right_on=['play', 'A', 'B'], how='left')
    per = []
    for w, g in m.dropna(subset=['weight_fr_alt']).groupby('work'):
        if len(g) >= 4:
            per.append((stats.spearmanr(g.weight_en, g.weight_fr)[0], stats.spearmanr(g.weight_en, g.weight_fr_alt)[0],
                        np.mean(np.sign(g.weight_en) == np.sign(g.weight_fr_alt))))
    per = np.array(per)
    edge_rows = pd.DataFrame([dict(version='original', mean_edge_rho=np.nanmean(per[:, 0])),
                              dict(version="without le/la/les/l'", mean_edge_rho=np.nanmean(per[:, 1]),
                                   sign_agreement=np.nanmean(per[:, 2]))])
    res.to_csv(OUT / 'french_articles.csv', index=False)
    edge_rows.to_csv(OUT / 'french_articles_edges.csv', index=False)
    print(f'\n== French articles without {sorted(FR_AMBIGUOUS_ARTICLES)} '
          f'(kept: {len(articles_fr)} entries)')
    print(res[['version', 'test', 'mean_1', 'mean_2', 'd', 'p']].round(4).to_string(index=False))
    print(f'matched edges with a French weight: {m.weight_fr_alt.notna().sum()} of {len(m)}')
    print(edge_rows.round(3).to_string(index=False))


def check_table(adj_pairs, n_shuffles):
    """Adjacent vs randomized by author / movement / overall, with the randomized side averaged over
    n_shuffles independent shuffles. p_perm = share of shuffles whose randomized mean reaches the
    adjacent mean (permutation test); p_t_median = median t-test p over shuffles."""
    adj = convergence(adj_pairs)
    groups = _dyad_index(adj_pairs)
    sel = {'Overall': lambda d: d}
    for a in ['Beckett', 'Ionesco', 'Genet', 'Becque', 'Brieux', 'Dumas']:
        sel[a] = (lambda a: lambda d: d[d.author == a])(a)
    for g in ['Absurdist', 'Realist']:
        sel[g] = (lambda g: lambda d: d[d.genre == g])(g)
    rows = []
    for s in range(n_shuffles):
        rnd = convergence(shuffle_replies(adj_pairs, 10_000 + s, groups))
        for name, f in sel.items():
            c = compare(f(adj)['conv'], f(rnd)['conv'])
            rows.append(dict(shuffle=s, group=name, adj_mean=c['mean_1'], rand_mean=c['mean_2'], d=c['d'], p=c['p']))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'shuffle_table_all.csv', index=False)
    summ = res.groupby('group', sort=False).agg(
        adj_mean=('adj_mean', 'first'), rand_mean=('rand_mean', 'mean'), d=('d', 'mean'),
        p_t_median=('p', 'median'), share_p05=('p', lambda x: (x < 0.05).mean()))
    summ['p_perm'] = res.groupby('group', sort=False).apply(
        lambda g: (1 + (g.rand_mean >= g.adj_mean).sum()) / (len(g) + 1))
    summ.to_csv(OUT / 'shuffle_table.csv')
    print(f'\n== Adjacent vs randomized, randomized averaged over {n_shuffles} shuffles')
    print(summ.round(4).to_string())


def check_counts():
    """Dyad-family scores retained after the thresholds, per play and condition (pipeline outputs)."""
    rows = []
    for play, lang in plays():
        base = f'outputs/plays_{lang}/wordlists/{play}/analysis_results/convergence_results/conv_by_pair_feature_wordlists'
        row = dict(play=play, lang=lang.upper())
        for name, suf in [('adjacent', ''), ('randomized', '_randomized'), ('nonadjacent', '_nonadjacent')]:
            row[name] = int(pd.read_csv(base + suf + '.csv')['conv'].notna().sum())
        rows.append(row)
    res = pd.DataFrame(rows)
    res.loc[len(res)] = dict(play='total', lang='', **res[['adjacent', 'randomized', 'nonadjacent']].sum())
    res.to_csv(OUT / 'retained_counts.csv', index=False)
    print('\n== Dyad-family scores retained after the thresholds')
    print(res.to_string(index=False))


def check_length():
    """Absurdist vs realist accommodation within comparable reply lengths: Mantel-Haenszel odds ratio of
    B using f after A used f, stratified by play x family x length bins of both turns."""
    d = pd.read_csv('outputs/alignment/reply_level.csv.gz')
    d = d[d.condition.isin(['adjacent', 'randomized'])]
    edges = [0, 4, 8, 15, 30, np.inf]
    d['b_bin'] = pd.cut(d.b_len, edges, right=True, labels=False)
    d['a_bin'] = pd.cut(d.a_len, edges, right=True, labels=False)

    def mh(g, strata):
        g = g.assign(n11=(g.a_uses == 1) & (g.b_uses == 1), n10=(g.a_uses == 1) & (g.b_uses == 0),
                     n01=(g.a_uses == 0) & (g.b_uses == 1), n00=(g.a_uses == 0) & (g.b_uses == 0))
        s = g.groupby(strata)[['n11', 'n10', 'n01', 'n00']].sum()
        n = s.sum(axis=1)
        return (s.n11 * s.n00 / n).sum() / (s.n10 * s.n01 / n).sum()

    rows = []
    for cond in ['adjacent', 'randomized']:
        c = d[d.condition == cond]
        for strata_name, strata in [('play x family', ['play', 'family']),
                                    ('play x family x length bins', ['play', 'family', 'a_bin', 'b_bin'])]:
            row = dict(condition=cond, strata=strata_name)
            for g in ['Absurdist', 'Realist']:
                row[g] = mh(c[c.genre == g], strata)
            rows.append(row)
        for b in range(len(edges) - 1):
            cb = c[c.b_bin == b]
            rows.append(dict(condition=cond, strata=f'reply length bin {edges[b]}-{edges[b + 1]}',
                             Absurdist=mh(cb[cb.genre == 'Absurdist'], ['play', 'family', 'a_bin']),
                             Realist=mh(cb[cb.genre == 'Realist'], ['play', 'family', 'a_bin']),
                             share_absurdist=(cb.genre == 'Absurdist').mean()))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'length_stratified.csv', index=False)
    print('\n== Absurdist vs realist odds ratios within reply-length strata')
    print(res.round(3).to_string(index=False))


def check_reciprocity():
    """Pearson (paper, from compute_stats log) vs Spearman (network_similarity descriptors) reciprocity."""
    import re
    log = open('outputs/logs/08_compute_stats.log', encoding='utf-8').read()
    pear = {m.group(1).strip(): float(m.group(2)) for m in
            re.finditer(r'^\s+(.+?): nodes=\d+.*?recip_corr=(-?[\d.]+|nan)', log, re.M)}
    desc = pd.read_csv('outputs/network_similarity/descriptors_conv.csv')
    names = list(pear)
    res = pd.DataFrame({'play': names, 'pearson': [pear[n] for n in names]})
    # log titles -> work id used in the descriptors file
    work_of = {
        'Waiting for Godot (EN)': 'Godot', 'En Attendant Godot (FR)': 'Godot',
        'Endgame (EN)': 'Endgame', 'Fin de Partie (FR)': 'Endgame',
        'The Balcony (EN)': 'Balcon', 'Le Balcon (FR)': 'Balcon',
        'The Maids (EN)': 'Bonnes', 'Les Bonnes (FR)': 'Bonnes',
        'The Bald Soprano (EN)': 'Cantatrice', 'La Cantatrice Chauve (FR)': 'Cantatrice',
        'Rhinoceros (EN)': 'Rhinoceros', 'Rhinoceros (FR)': 'Rhinoceros',
        'The Vultures (EN)': 'Corbeaux', 'Les Corbeaux (FR)': 'Corbeaux',
        'The Woman of Paris (EN)': 'Parisienne', 'La Parisienne (FR)': 'Parisienne',
        'The Red Robe (EN)': 'RobeRouge', 'La Robe Rouge (FR)': 'RobeRouge',
        'Woman on Her Own (EN)': 'FemmeSeule', 'La Femme Seule (FR)': 'FemmeSeule',
        'Camille (EN)': 'Camelias', 'La Dame aux Camelias (FR)': 'Camelias',
        'The Demi-Monde (EN)': 'DemiMonde', 'Le Demi-Monde (FR)': 'DemiMonde',
    }
    sp = {(r.work, r.language): r.recip_corr for r in desc.itertuples()}
    res['spearman'] = [sp.get((work_of[n], 'English' if n.endswith('(EN)') else 'French'), np.nan) for n in names]
    res['same_sign'] = np.sign(res.pearson) == np.sign(res.spearman)
    res.to_csv(OUT / 'reciprocity_pearson_spearman.csv', index=False)
    print('\n== Reciprocity: Pearson vs Spearman')
    print(res.round(2).to_string(index=False))
    print('same sign:', int(res.same_sign.sum()), 'of', len(res),
          '| rank correlation across plays:', round(stats.spearmanr(res.pearson, res.spearman, nan_policy='omit')[0], 2))


def check_response(adj_pairs, non_pairs):
    rows = []
    for name, mr in [('no response threshold (paper)', None), ('response threshold 2', 2)]:
        adj, non = convergence(adj_pairs, mr), convergence(non_pairs, mr)
        rnd = convergence(shuffle_replies(adj_pairs, 123), mr)
        for test, a, b in [
            ('adjacent vs randomized', adj.conv, rnd.conv),
            ('adjacent vs non-adjacent', adj.conv, non.conv),
            ('absurdist vs realist', adj.loc[adj.genre == 'Absurdist', 'conv'], adj.loc[adj.genre == 'Realist', 'conv']),
            ('English vs French', adj.loc[adj.lang == 'EN', 'conv'], adj.loc[adj.lang == 'FR', 'conv']),
        ]:
            rows.append(dict(version=name, test=test, **compare(a, b)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'response_filter.csv', index=False)
    print('\n== Response filter (B uses the family in >= 2 triggered replies)')
    print(res[['version', 'test', 'mean_1', 'mean_2', 'n_1', 'n_2', 'd', 'p']].round(4).to_string(index=False))


def check_lexicon():
    rows = [dict(family=FAMILIES[f], en=len(LISTS['en'][f]), fr=len(LISTS['fr'][f])) for f in PREFIXES]
    res = pd.DataFrame(rows)
    res.loc[len(res)] = dict(family='total', en=res.en.sum(), fr=res.fr.sum())
    res.to_csv(OUT / 'lexicon_sizes.csv', index=False)
    print('\n== Word list entries per family')
    print(res.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checks', nargs='*', default=['lexicon', 'response', 'addressee', 'french', 'seeds'])
    ap.add_argument('--n_seeds', type=int, default=50)
    ap.add_argument('--n_shuffles', type=int, default=1000, help='shuffles for the averaged table')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    adj_pairs = pd.concat([load_pairs(p, l) for p, l in plays()], ignore_index=True)
    check = convergence(adj_pairs)
    print(f'Recomputed adjacent mean {check.conv.mean():.5f} over {len(check)} scores '
          '(pipeline: 0.03070 over 5460)')
    if 'lexicon' in args.checks:
        check_lexicon()
    if {'addressee', 'response'} & set(args.checks):
        non_pairs = pd.concat([load_pairs(p, l, '_nonadjacent') for p, l in plays()], ignore_index=True)
    if 'response' in args.checks:
        check_response(adj_pairs, non_pairs)
    if 'addressee' in args.checks:
        check_addressee(adj_pairs, non_pairs)
    if 'french' in args.checks:
        check_french(adj_pairs)
    if 'seeds' in args.checks:
        check_seeds(adj_pairs, args.n_seeds)
    if 'table' in args.checks:
        check_table(adj_pairs, args.n_shuffles)
    if 'counts' in args.checks:
        check_counts()
    if 'length' in args.checks:
        check_length()
    if 'reciprocity' in args.checks:
        check_reciprocity()


if __name__ == '__main__':
    main()
