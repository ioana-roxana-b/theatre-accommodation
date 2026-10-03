"""
Generate enhanced LaTeX table for mixed effects model with standardized coefficients.
"""
import pandas as pd
import numpy as np

# Load the fixed effects results
fe = pd.read_csv('outputs/mixed_effects_fixed_effects.csv', index_col=0)

# Remove NaN rows (Group Var)
fe = fe.dropna(subset=['coef_raw'])

# Sort by absolute standardized coefficient for prominence
fe['abs_std'] = fe['coef_std'].abs()
fe_sorted = fe.sort_values('abs_std', ascending=False)

# Build LaTeX table
latex_lines = []

latex_lines.append(r"\begin{table}[htbp]")
latex_lines.append(r"\centering")
latex_lines.append(r"\small")
latex_lines.append(r"\caption{Mixed Effects Model: Raw and Standardized Coefficients.}")
latex_lines.append(r"\label{tab:mixed_effects_std}")
latex_lines.append(r"\begin{tabular}{lrrrrrr}")
latex_lines.append(r"\toprule")
latex_lines.append(r"Factor & \multicolumn{3}{c}{Raw Scale} & \multicolumn{3}{c}{Standardized Scale} \\")
latex_lines.append(r"       & $\beta$ & SE & $p$ & $\beta_{\text{std}}$ & SE & $p$ \\")
latex_lines.append(r"\midrule")

# Process each row
for idx, row in fe_sorted.iterrows():
    # Skip intercept for now, will add at top
    if 'Intercept' in idx:
        continue

    # Format factor name
    if 'feature_family' in idx:
        var_group = 'Feature family'
        var_name = idx.split('[T.')[-1].replace(']', '')
    elif 'genre' in idx:
        var_group = 'Genre'
        var_name = idx.split('[T.')[-1].replace(']', '')
    elif 'language' in idx:
        var_group = 'Language'
        var_name = idx.split('[T.')[-1].replace(']', '')
    elif 'gender_dyad' in idx:
        var_group = 'Gender dyad'
        var_name = idx.split('[T.')[-1].replace(']', '')
    elif 'importance' in idx:
        var_group = 'Importance'
        var_name = idx.split('[T.')[-1].replace(']', '')
    else:
        continue

    # Format p-value
    p = row['p']
    if p < 0.0001:
        p_str = "$<$.0001$^{***}$"
    elif p < 0.001:
        p_str = f"${p:.4f}^{{***}}$"
    elif p < 0.01:
        p_str = f"${p:.4f}^{{**}}$"
    elif p < 0.05:
        p_str = f"${p:.4f}^*$"
    else:
        p_str = f"${p:.4f}$"

    # Format effect size indicator
    abs_std = abs(row['coef_std'])
    if abs_std > 0.20:
        effect_marker = r"$\dagger$"  # Large
    elif abs_std > 0.10:
        effect_marker = ""
    else:
        effect_marker = ""

    # Output row
    latex_lines.append(f"  {var_name:30s} & {row['coef_raw']:+.4f} & {row['se_raw']:.4f} & {p_str} & {row['coef_std']:+.4f}{effect_marker} & {row['se_std']:.4f} & {p_str} \\\\")

latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
latex_lines.append(r"\note{")
latex_lines.append(r"N = 5,363 observations from 24 plays. ")
latex_lines.append(r"Raw scale: original convergence score units. ")
latex_lines.append(r"Standardized scale: convergence score $z$-scored (M=0, SD=1). ")
latex_lines.append(r"$^*p<.05$, $^{**}p<.01$, $^{***}p<.001$. ")
latex_lines.append(r"$\dagger$ indicates large effect ($|\beta_{\text{std}}| > 0.20$). ")
latex_lines.append(r"}")
latex_lines.append(r"\end{table}")

# Write to file
latex_output = "\n".join(latex_lines)

with open('paper/tables/mixed_effects_latex_table_enhanced.tex', 'w', encoding='utf-8') as f:
    f.write(latex_output)

print(latex_output)
print("\n\nLaTeX table saved to: mixed_effects_latex_table_enhanced.tex")

# Also generate relative importance table
print("\n" + "="*80)
print("RELATIVE IMPORTANCE TABLE (Ranked by Standardized Effect Size)")
print("="*80)

importance_lines = []
importance_lines.append(r"\begin{table}[htbp]")
importance_lines.append(r"\centering")
importance_lines.append(r"\small")
importance_lines.append(r"\caption{Relative Importance of Predictors (Rank-Ordered).}")
importance_lines.append(r"\label{tab:relative_importance}")
importance_lines.append(r"\begin{tabular}{rllrr}")
importance_lines.append(r"\toprule")
importance_lines.append(r"Rank & Factor & Category & $|\beta_{\text{std}}|$ & Effect Size \\")
importance_lines.append(r"\midrule")

for i, (idx, row) in enumerate(fe_sorted.iterrows(), 1):
    if 'Intercept' in idx or pd.isna(row['coef_std']):
        continue

    # Parse factor
    if 'feature_family' in idx:
        category = 'Word family'
        name = idx.split('[T.')[-1].replace(']', '')
    elif 'importance' in idx:
        category = 'Character importance'
        name = idx.split('[T.')[-1].replace(']', '')
    elif 'genre' in idx:
        category = 'Genre'
        name = idx.split('[T.')[-1].replace(']', '')
    elif 'language' in idx:
        category = 'Language'
        name = idx.split('[T.')[-1].replace(']', '')
    elif 'gender_dyad' in idx:
        category = 'Gender'
        name = idx.split('[T.')[-1].replace(']', '')
    else:
        continue

    abs_coef = abs(row['coef_std'])
    if abs_coef > 0.20:
        effect = "Large"
    elif abs_coef > 0.10:
        effect = "Medium"
    elif abs_coef > 0.05:
        effect = "Small"
    else:
        effect = "Negligible"

    p_sig = "***" if row['p'] < 0.001 else ("**" if row['p'] < 0.01 else ("*" if row['p'] < 0.05 else ""))

    importance_lines.append(f"{i} & {name} & {category} & {abs_coef:.3f}{p_sig} & {effect} \\\\")

importance_lines.append(r"\bottomrule")
importance_lines.append(r"\end{tabular}")
importance_lines.append(r"\note{")
importance_lines.append(r"Predictors ranked by absolute value of standardized coefficient. ")
importance_lines.append(r"Effect sizes: Large $|\beta_{\text{std}}| > 0.20$; Medium 0.10--0.20; Small 0.05--0.10; Negligible $<0.05$. ")
importance_lines.append(r"}")
importance_lines.append(r"\end{table}")

importance_output = "\n".join(importance_lines)

with open('paper/tables/mixed_effects_relative_importance.tex', 'w', encoding='utf-8') as f:
    f.write(importance_output)

print("\n" + importance_output)
print("\n\nRelative importance table saved to: mixed_effects_relative_importance.tex")
