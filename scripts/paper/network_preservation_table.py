"""
LaTeX table for the cross-language network preservation analysis
(scripts/stats/network_preservation.py, run that first).

Writes paper/tables/network_preservation.tex: per work, how well the EN and FR accommodation
networks agree (edge-weight rho, sign agreement vs chance, character out-strength rho), for the
DNM convergence weights used in the paper's networks, with the pooled permutation tests for those
and, as a robustness row, for the trigger weights (not scaled by the family's base rate).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stats'))
from build_alignment_table import ABSURDIST, WORKS  # noqa: E402

IN = 'outputs/network_preservation'
TITLES = {
    'Godot': 'Waiting for Godot', 'Endgame': 'Endgame', 'Corbeaux': 'The Vultures',
    'Parisienne': 'The Woman of Paris', 'RobeRouge': 'The Red Robe', 'FemmeSeule': 'Woman on Her Own',
    'Camelias': 'Camille', 'DemiMonde': 'The Demi-Monde', 'Balcon': 'The Balcony', 'Bonnes': 'The Maids',
    'Cantatrice': 'The Bald Soprano', 'Rhinoceros': 'Rhinoceros',
}


def stars(p):
    return '$^{***}$' if p < .001 else '$^{**}$' if p < .01 else '$^{*}$' if p < .05 else ''


def num(x, p=None):
    if pd.isna(x):
        return '--'
    s = f'{x:.2f}'.replace('-', '$-$')
    return s + (stars(p) if p is not None and pd.notna(p) else '')


def fmt_p(p):
    return '$<$.001' if p < .001 else f'{p:.3f}'.lstrip('0')


pw = pd.read_csv(f'{IN}/per_work_conv.csv').set_index('work')
pooled = {w: pd.read_csv(f'{IN}/pooled_{w}.csv').set_index('measure') for w in ['conv', 'trigger']}

L = [r'\begin{table*}[htbp]', r'\centering', r'\small',
     r'\caption{Cross-language preservation of accommodation networks. For each work, directed edges '
     r'$A\to B$ present in both versions are compared (characters aligned across versions). '
     r'Edge weight: mean convergence $\overline{\mathrm{Conv}}_{A \to B}$ over the nine families, as in the '
     r'network analysis. $p$-values from permutation within work.}',
     r'\label{tab:network_preservation}',
     r'\begin{tabular}{llrrrrr}', r'\toprule',
     r'Play & Movement & Edges & Edge $\rho$ & Same sign (chance) & Out-strength $\rho$ & In-strength $\rho$ \\',
     r'\midrule']
for movement in ['Absurdist', 'Realist']:
    for work, (en_play, _) in WORKS.items():
        is_abs = en_play.split('_')[0] in ABSURDIST
        if (movement == 'Absurdist') != is_abs:
            continue
        r = pw.loc[work]
        L.append(f"\\textit{{{TITLES[work]}}} & {movement} & {int(r['matched_edges'])} & "
                 f"{num(r['rho'], r['rho_p'])} & {r['sign_agree']:.0%} ({r['sign_expected']:.0%}){stars(r['sign_p'])} & "
                 f"{num(r.get('out_strength_rho'), r.get('out_strength_p'))} & "
                 f"{num(r.get('in_strength_rho'), r.get('in_strength_p'))} \\\\".replace('%', r'\%'))
    L.append(r'\midrule')
for w, label in [('conv', 'Pooled'), ('trigger', 'Pooled, trigger weights')]:
    rho_row, sign_row = pooled[w].loc['mean edge rho'], pooled[w].loc['sign agreement']
    L.append(f"\\textit{{{label}}} & & {int(rho_row['edges'])} & {num(rho_row['observed'])} ($p$ {'' if rho_row['p'] < .001 else '= '}{fmt_p(rho_row['p'])}) & "
             f"{sign_row['observed']:.0%} ({sign_row['null_mean']:.0%}; $p$ {'' if sign_row['p'] < .001 else '= '}{fmt_p(sign_row['p'])}) & & \\\\".replace('%', r'\%'))
L += [r'\bottomrule', r'\end{tabular}',
      r'\\[2pt]{\footnotesize Pooled edge $\rho$: mean of per-work Spearman correlations; null: FR weights '
      r'shuffled across matched edges within every work. Trigger weights (robustness): '
      r'$P(B \mid A \text{ used } f) - P(B \mid A \text{ did not})$, not scaled by how often A uses $f$. '
      r'$^*p<.05$, $^{**}p<.01$, $^{***}p<.001$.}',
      r'\end{table*}']
with open('paper/tables/network_preservation.tex', 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('Saved paper/tables/network_preservation.tex')

# ── Cross-play similarity (scripts/stats/network_similarity.py) ────────────
SIM = 'outputs/network_similarity'
if Path(f'{SIM}/summary_trigger.csv').exists():
    L = [r'\begin{table}[htbp]', r'\centering', r'\small',
         r'\caption{Is a play\'s French network closer to its own English version than to other plays? '
         r'Networks compared through descriptors of how accommodation is organised (no character alignment). '
         r'Matching null: French networks re-assigned to random works. Topology null: weights shuffled within '
         r'each network, keeping cast and who-talks-to-whom.}'.replace("\\'", "'"),
         r'\label{tab:network_similarity}',
         r'\begin{tabular}{llrrrr}', r'\toprule',
         r'Weights & Statistic & Observed & $p$ (matching) & Topology null & $p$ (topology) \\', r'\midrule']
    for w, label in [('conv', 'Convergence'), ('trigger', 'Trigger')]:
        s = pd.read_csv(f'{SIM}/summary_{w}.csv')
        for i, r in s.iterrows():
            stat = 'Matched advantage' if r['statistic'] == 'diag_advantage' else 'Own version nearest (of 12)'
            obs = f"{r['observed']:.2f}" if i == 0 else f"{int(r['observed'])}"
            null = f"{r['topology_null_mean']:.2f}" if i == 0 else f"{r['topology_null_mean']:.1f}"
            L.append(f"{label if i == 0 else ''} & {stat} & {obs} & {fmt_p(r['matching_p'])}{stars(r['matching_p'])} & "
                     f"{null} & {fmt_p(r['topology_p'])}{stars(r['topology_p'])} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open('paper/tables/network_similarity.tex', 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('Saved paper/tables/network_similarity.tex')
