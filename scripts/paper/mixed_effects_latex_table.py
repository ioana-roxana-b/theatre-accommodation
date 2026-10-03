"""
Generate LaTeX tables for the mixed effects model, reading the CSVs written by
scripts/stats/mixed_effects_model.py (run that first).

Writes:
  paper/tables/mixed_effects_latex_table.tex   fixed effects (raw scale) + random effects
  paper/tables/mixed_effects_lrt.tex           likelihood-ratio test per factor
  paper/tables/mixed_effects_genre_robustness.tex  genre effect, primary vs nested model
"""
import pandas as pd

fe = pd.read_csv('outputs/mixed_effects_fixed_effects.csv', index_col=0)
lrt = pd.read_csv('outputs/mixed_effects_lrt_importance.csv')
nested = pd.read_csv('outputs/mixed_effects_nested_robustness.csv', index_col=0)
info = pd.read_csv('outputs/mixed_effects_model_info.csv', index_col=0)['value']

# (formula term, label, reference level)
FACTORS = [
    ('genre',               'Genre',          'Absurdist'),
    ('language',            'Language',       'English'),
    ('gender_dyad',         'Gender dyad',    'F-F'),
    ('importance_dyad_dir', 'Importance (A$\\to$B)', 'intermediate$\\to$intermediate'),
    ('feature_family',      'Feature family', 'adverbs'),
]


def stars(p):
    return '$^{***}$' if p < .001 else '$^{**}$' if p < .01 else '$^{*}$' if p < .05 else ''


def fmt_p(p):
    if p < .0001:
        return '$<$.0001' + stars(p)
    if p < .001:
        return '$<$.001' + stars(p)
    return f'{p:.3f}'.lstrip('0') + stars(p)


def fmt_num(x, nd=3):
    s = f'{x:.{nd}f}'
    return s.replace('-', '$-$')


def level_name(idx):
    return idx.split('[T.')[-1].rstrip(']').replace('->', '$\\to$').replace('_', ' ')


def write(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved {path}')


# ── Fixed effects table ────────────────────────────────────────────────────
L = [r'\begin{table}[htbp]', r'\centering', r'\small',
     r'\caption{Mixed effects model of convergence score.}',
     r'\label{tab:mixed_effects}',
     r'\begin{tabular}{lrrr}', r'\toprule',
     r'Factor & $\beta$ & SE & $p$ \\', r'\midrule']
row = fe.loc['Intercept']
L.append(f"Intercept & {fmt_num(row['coef_raw'])} & {row['se_raw']:.3f} & {fmt_p(row['p'])} \\\\")
for term, label, ref in FACTORS:
    rows = fe[fe.index.str.startswith(f'C({term})')]
    if term == 'feature_family':
        rows = rows.sort_values('coef_raw', ascending=False)
    L.append(f'\\textit{{{label}}} (ref.: {ref}) \\\\')
    for idx, r in rows.iterrows():
        L.append(f"\\quad {level_name(idx)} & {fmt_num(r['coef_raw'])} & {r['se_raw']:.3f} & {fmt_p(r['p'])} \\\\")
L.append(r'\midrule')
L.append(r'\multicolumn{4}{l}{\textit{Random effects (SD)}} \\')
for key, label in [('sd_play', 'Play'), ('sd_dyad', 'Dyad'), ('sd_residual', 'Residual')]:
    if key in info and pd.notna(info[key]):
        L.append(f'\\quad {label} & {info[key]:.3f} & & \\\\')
L += [r'\bottomrule', r'\end{tabular}',
      f"\\\\[2pt]{{\\footnotesize N = {int(info['n_obs']):,} dyad $\\times$ family observations, "
      f"{int(info['n_dyads']):,} dyads, {int(info['n_plays'])} plays. "
      r'$^*p<.05$, $^{**}p<.01$, $^{***}p<.001$.}',
      r'\end{table}']
write('paper/tables/mixed_effects_latex_table.tex', L)

# ── Likelihood-ratio tests ─────────────────────────────────────────────────
labels = {t: lab for t, lab, _ in FACTORS}
L = [r'\begin{table}[htbp]', r'\centering', r'\small',
     r'\caption{Likelihood-ratio tests: each factor dropped from the full model (ML fit).}',
     r'\label{tab:mixed_effects_lrt}',
     r'\begin{tabular}{lrrr}', r'\toprule',
     r'Factor & df & $\chi^2$ & $p$ \\', r'\midrule']
for _, r in lrt.sort_values('chi2', ascending=False).iterrows():
    p = '$<$.0001' + stars(r['p']) if r['p'] < .0001 else fmt_p(r['p'])
    L.append(f"{labels.get(r['factor'], r['factor'])} & {int(r['df'])} & {r['chi2']:.2f} & {p} \\\\")
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
write('paper/tables/mixed_effects_lrt.tex', L)

# ── Genre robustness: primary vs nested (play within author) ───────────────
g = 'C(genre)[T.Realist]'
L = [r'\begin{table}[htbp]', r'\centering', r'\small',
     r'\caption{Genre effect (Realist vs.\ Absurdist) under two random-effects structures.}',
     r'\label{tab:genre_robustness}',
     r'\begin{tabular}{lrrr}', r'\toprule',
     r'Model & $\beta$ & SE & $p$ \\', r'\midrule',
     f"Primary (random intercept: play) & {fmt_num(fe.loc[g, 'coef_raw'])} & {fe.loc[g, 'se_raw']:.3f} & {fmt_p(fe.loc[g, 'p'])} \\\\",
     f"Robustness (random intercept: author; play nested) & {fmt_num(nested.loc[g, 'coef'])} & {nested.loc[g, 'se']:.3f} & {fmt_p(nested.loc[g, 'p'])} \\\\",
     r'\bottomrule', r'\end{tabular}', r'\end{table}']
write('paper/tables/mixed_effects_genre_robustness.tex', L)
