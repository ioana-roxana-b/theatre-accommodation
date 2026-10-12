from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path('outputs/turn_length')
LONG = 50


def dnm(g):
    return g.loc[g.a_uses == 1, 'b_uses'].mean() - g.b_uses.mean()


def ols(t, cols):
    X = np.column_stack([np.ones(len(t))] + [t[c].to_numpy(float) for c in cols])
    y = t.conv.to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    df = len(y) - X.shape[1]
    se = np.sqrt(np.diag(e @ e / df * np.linalg.inv(X.T @ X)))
    p = 2 * stats.t.sf(np.abs(b / se), df)
    r2 = 1 - e @ e / ((y - y.mean()) @ (y - y.mean()))
    model = ' + '.join(cols)
    return [{'test': f'OLS conv ~ {model}', 'term': c, 'estimate': bb, 'p': pp, 'r2': r2}
            for c, bb, pp in zip(cols, b[1:], p[1:])]


def dyad_mean_conv(play, language):
    """Mean of the play's adjacent dyad-family scores, the per-play Conv of the paper's Table 6."""
    lang = 'en' if language == 'English' else 'fr'
    f = Path(f'outputs/plays_{lang}/wordlists/{play}/analysis_results/convergence_results/'
             'conv_by_pair_feature_wordlists.csv')
    return pd.read_csv(f)['conv'].dropna().mean()


def main():
    d = pd.read_csv('outputs/alignment/reply_level.csv.gz')
    d = d[d.condition == 'adjacent']
    pooled = (d.groupby(['play', 'family'])[['a_uses', 'b_uses']].apply(dnm)
              .groupby('play').mean().rename('conv_pooled'))
    replies = d.drop_duplicates(['play', 'pair_id'])
    t = replies.groupby('play').agg(
        author=('author', 'first'), genre=('genre', 'first'), language=('language', 'first'),
        n_replies=('b_len', 'size'), mean_len=('b_len', 'mean'), median_len=('b_len', 'median'),
        long_share=('b_len', lambda x: (x >= LONG).mean())).join(pooled)
    t['conv'] = [dyad_mean_conv(p, lang) for p, lang in zip(t.index, t.language)]
    t['log_mean_len'] = np.log(t.mean_len)
    t['absurdist'] = (t.genre == 'Absurdist').astype(int)

    tests = []
    for measure in ['conv', 'conv_pooled']:
        for col in ['mean_len', 'median_len', 'long_share']:
            for scope, s in [('all', t)] + list(t.groupby('genre')):
                r, p = stats.spearmanr(s[col], s[measure])
                tests.append({'test': f'Spearman {col} vs {measure} ({scope}, n={len(s)})', 'term': col,
                              'estimate': r, 'p': p, 'r2': np.nan})
    for cols in (['absurdist'], ['log_mean_len'], ['absurdist', 'log_mean_len']):
        tests += ols(t, cols)
    tests = pd.DataFrame(tests)

    OUT.mkdir(parents=True, exist_ok=True)
    t.sort_values('conv').round(4).to_csv(OUT / 'turn_length_by_play.csv')
    tests.round(4).to_csv(OUT / 'turn_length_tests.csv', index=False)

    pd.set_option('display.width', 200)
    print(t.sort_values('conv')[['genre', 'language', 'mean_len', 'median_len', 'long_share', 'conv',
                                 'conv_pooled']].round(3))
    print()
    print(t.groupby('genre')[['mean_len', 'median_len', 'long_share', 'conv', 'conv_pooled']].mean().round(3))
    print(t.groupby('author')[['mean_len', 'conv']].mean().round(3))
    print()
    print(tests.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
