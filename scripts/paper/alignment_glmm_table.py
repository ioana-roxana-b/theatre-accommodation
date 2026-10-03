"""
LaTeX tables for the reply-level logistic mixed model (scripts/stats/alignment_glmm.R, run that first).

Writes:
  paper/tables/alignment_glmm_lrt.tex     does accommodation differ by each factor (LRT on a_uses:<factor>)
  paper/tables/alignment_glmm_levels.tex  accommodation (odds ratio of a_uses) per level of each factor
  paper/tables/alignment_glmm_genre_family.tex  absurdist vs realist accommodation per family
  paper/tables/alignment_glmm_conditions.tex    adjacent vs non-adjacent vs randomized (if alignment_conditions.R was run)
"""
import math
import os

import pandas as pd

OUT = 'outputs/alignment'
lrt = pd.read_csv(f'{OUT}/glmm_lrt.csv')
acc = pd.read_csv(f'{OUT}/glmm_accommodation_by_level.csv')
info = dict(l.strip().split(',') for l in open(f'{OUT}/glmm_model_info.csv'))

LABELS = {
    'genre': 'Genre', 'language': 'Language', 'family': 'Feature family',
    'importance_dyad': 'Importance (A$\\to$B)', 'gender_dyad': 'Gender dyad',
}


def stars(p):
    return '$^{***}$' if p < .001 else '$^{**}$' if p < .01 else '$^{*}$' if p < .05 else ''


def fmt_p(p):
    if p < .0001:
        return '$<$.0001' + stars(p)
    if p < .001:
        return '$<$.001' + stars(p)
    return f'{p:.3f}'.lstrip('0') + stars(p)


def tex(s):
    return str(s).replace('->', '$\\to$').replace('_', ' ')


def write(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved {path}')


note = (f"{int(info['n_pairs']):,} adjacent pairs $\\times$ 9 families, {int(info['n_dyads']):,} dyads, "
        f"{info['n_plays']} plays ({info['n_works']} works).")

# ── LRT table ──────────────────────────────────────────────────────────────
L = [r'\begin{table}[htbp]', r'\centering', r'\small',
     r'\caption{Does accommodation differ by factor? Likelihood-ratio tests on the '
     r'interaction of each factor with whether A used the family.}',
     r'\label{tab:glmm_lrt}',
     r'\begin{tabular}{lrrr}', r'\toprule',
     r'Interaction & df & $\chi^2$ & $p$ \\', r'\midrule']
for _, r in lrt.iterrows():
    factor = r['test'].replace('a_uses:', '')
    label = 'Genre $\\times$ family' if factor == 'genre:family' else LABELS.get(factor, tex(factor))
    L.append(f"{label} & {int(r['df'])} & {r['chi2']:.2f} & {fmt_p(r['p'])} \\\\")
L += [r'\bottomrule', r'\end{tabular}', f'\\\\[2pt]{{\\footnotesize {note}}}', r'\end{table}']
write('paper/tables/alignment_glmm_lrt.tex', L)

# ── Accommodation per level ────────────────────────────────────────────────
L = [r'\begin{table}[htbp]', r'\centering', r'\small',
     r'\caption{Accommodation by level: odds ratio of B using a family when A just used it '
     r'(other factors at their reference level).}',
     r'\label{tab:glmm_levels}',
     r'\begin{tabular}{lrrr}', r'\toprule',
     r'Level & OR & 95\% CI & $p$ \\', r'\midrule']
for factor, grp in acc.groupby('factor', sort=False):
    L.append(f'\\textit{{{LABELS.get(factor, tex(factor))}}} \\\\')
    if factor == 'family':
        grp = grp.sort_values('estimate', ascending=False)
    for _, r in grp.iterrows():
        lo, hi = r['estimate'] - 1.96 * r['se'], r['estimate'] + 1.96 * r['se']
        L.append(f"\\quad {tex(r['level'])} & {r['odds_ratio']:.2f} & "
                 f"[{math.exp(lo):.2f}, {math.exp(hi):.2f}] & {fmt_p(r['p'])} \\\\")
L += [r'\bottomrule', r'\end{tabular}', f'\\\\[2pt]{{\\footnotesize {note}}}', r'\end{table}']
write('paper/tables/alignment_glmm_levels.tex', L)

# ── Genre x family ─────────────────────────────────────────────────────────
gf = pd.read_csv(f'{OUT}/glmm_genre_by_family.csv').sort_values('diff_realist_minus_absurdist')
L = [r'\begin{table}[htbp]', r'\centering', r'\small',
     r'\caption{Accommodation (log-odds effect of A using the family) by genre and family, '
     r'from the model with the genre $\times$ family interaction (English, other factors at reference).}',
     r'\label{tab:glmm_genre_family}',
     r'\begin{tabular}{lrrrr}', r'\toprule',
     r'Family & Absurdist & Realist & Difference (SE) & $p$ \\', r'\midrule']
for _, r in gf.iterrows():
    L.append(f"{tex(r['family'])} & {r['absurdist']:.2f} & {r['realist']:.2f} & "
             f"{r['diff_realist_minus_absurdist']:+.2f} ({r['diff_se']:.2f}) & {fmt_p(r['diff_p'])} \\\\".replace('-', '$-$'))
L += [r'\bottomrule', r'\end{tabular}', f'\\\\[2pt]{{\\footnotesize {note}}}', r'\end{table}']
write('paper/tables/alignment_glmm_genre_family.tex', L)

# ── Adjacent vs control conditions (scripts/stats/alignment_conditions.R) ──
if os.path.exists(f'{OUT}/conditions_slopes.csv'):
    sl = pd.read_csv(f'{OUT}/conditions_slopes.csv')
    df_ = pd.read_csv(f'{OUT}/conditions_differences.csv')
    clrt = pd.read_csv(f'{OUT}/conditions_lrt.csv').iloc[0]
    orr = sl.pivot(index='family', columns='condition', values='odds_ratio')
    pv = df_.pivot(index='family', columns='comparison', values='p')
    order = ['all families'] + sorted(f for f in orr.index if f != 'all families')
    L = [r'\begin{table}[htbp]', r'\centering', r'\small',
         r'\caption{Accommodation (odds ratio of B using a family when A just used it) for adjacent, '
         r'non-adjacent and randomized pairs, and tests of adjacent against each control.}',
         r'\label{tab:glmm_conditions}',
         r'\begin{tabular}{lrrrrr}', r'\toprule',
         r' & \multicolumn{3}{c}{Odds ratio} & \multicolumn{2}{c}{$p$ (adjacent vs.)} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}',
         r'Family & Adj. & Non-adj. & Rand. & Non-adj. & Rand. \\', r'\midrule']
    for fam in order:
        name = r'\textit{All families}' if fam == 'all families' else tex(fam)
        L.append(f"{name} & {orr.loc[fam, 'adjacent']:.2f} & {orr.loc[fam, 'nonadjacent']:.2f} & "
                 f"{orr.loc[fam, 'randomized']:.2f} & {fmt_p(pv.loc[fam, 'adjacent - nonadjacent'])} & "
                 f"{fmt_p(pv.loc[fam, 'adjacent - randomized'])} \\\\")
        if fam == 'all families':
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}',
          f"\\\\[2pt]{{\\footnotesize Condition $\\times$ accommodation: $\\chi^2({int(clrt['df'])})$ = "
          f"{clrt['chi2']:.2f}, $p$ {fmt_p(clrt['p']) if clrt['p'] < .001 else '= ' + fmt_p(clrt['p'])}. Family averages are the mean of the per-family log-odds effects.}}",
          r'\end{table}']
    write('paper/tables/alignment_glmm_conditions.tex', L)
