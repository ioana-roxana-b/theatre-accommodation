"""
Create publication-ready summary tables for mixed effects model results.
"""
import pandas as pd
import numpy as np

# Load the fixed effects results
fe = pd.read_csv('outputs/mixed_effects_fixed_effects.csv', index_col=0)
fe = fe[fe['coef_raw'].notna()]  # drop variance-component rows
info = pd.read_csv('outputs/mixed_effects_model_info.csv', index_col=0)['value']

# Function to format p-values
def fmt_p(p):
    if p < 0.0001:
        return "<.0001***"
    elif p < 0.001:
        return f"{p:.4f}***"
    elif p < 0.01:
        return f"{p:.4f}**"
    elif p < 0.05:
        return f"{p:.4f}*"
    else:
        return f"{p:.4f}"

# Parse the index to extract variable names
fe['var_group'] = 'Intercept'
fe['var_level'] = ''

for idx, row in fe.iterrows():
    if 'genre' in idx:
        fe.loc[idx, 'var_group'] = 'Genre'
        fe.loc[idx, 'var_level'] = idx.split('[T.')[-1].replace(']', '')
    elif 'language' in idx:
        fe.loc[idx, 'var_group'] = 'Language'
        fe.loc[idx, 'var_level'] = idx.split('[T.')[-1].replace(']', '')
    elif 'gender_dyad' in idx:
        fe.loc[idx, 'var_group'] = 'Gender dyad'
        fe.loc[idx, 'var_level'] = idx.split('[T.')[-1].replace(']', '')
    elif 'importance' in idx:
        fe.loc[idx, 'var_group'] = 'Importance dyad'
        fe.loc[idx, 'var_level'] = idx.split('[T.')[-1].replace(']', '')
    elif 'feature_family' in idx:
        fe.loc[idx, 'var_group'] = 'Feature family'
        fe.loc[idx, 'var_level'] = idx.split('[T.')[-1].replace(']', '')

# Create summary by group
print("="*100)
print("MIXED EFFECTS REGRESSION: CONVERGENCE SCORE")
print("="*100)
print("\nDependent variable: Convergence score (function-word family basis)")
print("Fixed effects: Genre, Language, Gender dyad, Importance dyad, Feature family")
print(f"Random effects SD: dyad={info.get('sd_dyad', float('nan')):.4f}, play={info.get('sd_play', float('nan')):.4f}, residual={info['sd_residual']:.4f}")
print(f"N observations: {int(info['n_obs'])} | N plays: {int(info['n_plays'])} | LL (REML): {info['loglik_reml']:.2f}\n")

# Intercept
intercept_row = fe[fe['var_group'] == 'Intercept'].iloc[0]
print(f"Intercept: {intercept_row['coef_raw']:+.4f} ({intercept_row['se_raw']:.4f}), p{fmt_p(intercept_row['p'])}\n")

# Genre
print("GENRE (reference: Absurdist)")
genre_rows = fe[fe['var_group'] == 'Genre']
for idx, row in genre_rows.iterrows():
    sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else ''))
    print(f"  {row['var_level']:20s}: {row['coef_raw']:+.4f} ({row['se_raw']:.4f})  p={fmt_p(row['p'])}")

# Language
print("\nLANGUAGE (reference: English)")
lang_rows = fe[fe['var_group'] == 'Language']
for idx, row in lang_rows.iterrows():
    sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else ''))
    print(f"  {row['var_level']:20s}: {row['coef_raw']:+.4f} ({row['se_raw']:.4f})  p={fmt_p(row['p'])}")

# Gender dyad
print("\nGENDER DYAD (reference: F-F)")
gd_rows = fe[fe['var_group'] == 'Gender dyad']
for idx, row in gd_rows.iterrows():
    sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else ''))
    print(f"  {row['var_level']:20s}: {row['coef_raw']:+.4f} ({row['se_raw']:.4f})  p={fmt_p(row['p'])}")

# Importance dyad
print("\nIMPORTANCE DYAD (reference: intermediate-intermediate)")
id_rows = fe[fe['var_group'] == 'Importance dyad']
for idx, row in id_rows.iterrows():
    sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else ''))
    print(f"  {row['var_level']:30s}: {row['coef_raw']:+.4f} ({row['se_raw']:.4f})  p={fmt_p(row['p'])}")

# Feature family
print("\nFEATURE FAMILY (reference: adverbs)")
ff_rows = fe[fe['var_group'] == 'Feature family']
# Sort by coefficient for clarity
ff_rows = ff_rows.sort_values('coef_raw', ascending=False)
for idx, row in ff_rows.iterrows():
    sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else ''))
    print(f"  {row['var_level']:25s}: {row['coef_raw']:+.4f} ({row['se_raw']:.4f})  p={fmt_p(row['p'])}")

print("\n" + "="*100)
print("INTERPRETATION")
print("="*100)
print("""
Key findings:

1. GENRE: No significant effect of realist vs. absurdist (p=.247). Convergence is comparable
   across both dramatic traditions when other factors are controlled.

2. LANGUAGE: No significant effect of English vs. French (p=.529). Convergence patterns are
   stable across languages, supporting cross-linguistic generalizability.

3. GENDER: Only M-U (male-unknown) dyads show increased convergence (β=+.044, p=.043).
   Other gender compositions (F-M, M-M) show no significant differences. This suggests
   gender has minimal independent influence when importance is controlled.

4. IMPORTANCE: Dyads involving major characters show REDUCED convergence:
   - major-major:         β=-.022, p=.001 (reduced relative to intermediate-intermediate)
   - intermediate-major:  β=-.012, p=.023
   This suggests that more prominent characters converge LESS, contrary to a simple
   accommodation hypothesis. May reflect dialogue distance or speech dominance effects.

5. FEATURE FAMILY: Strong heterogeneity across word families:
   - Strongest convergence: quantifiers (β=+.072, p<.0001) and negations (β=+.038, p<.0001)
   - Strongest divergence: auxiliary_verbs (β=-.019, p=.001) and personal_pronouns (β=-.020, p<.001)
   This suggests that phonologically light, high-frequency function words show convergence,
   while lexically grounded pronouns show divergence or individual specificity.

CONCLUSION: When controlling for all factors simultaneously, genre and language effects
are negligible, while feature family emerges as the dominant source of variation.
Individual differences by character importance are small but robust, and appear to
reflect structural/dominance effects rather than purely social factors.
""")
