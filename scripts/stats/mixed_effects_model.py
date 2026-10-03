"""
Mixed effects regression model for convergence analysis.

Fits a single multivariate model with convergence score as DV and:
  - Fixed effects: genre, language, gender dyad type, directed importance dyad, feature family
  - Random intercepts: play (primary model), with a dyad-level variance component
    to account for repeated observations of the same A->B dyad across feature families
  - Robustness check: nested random effects (play within author) to test whether the
    genre effect survives a stricter, author-level grouping
"""
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path
from statsmodels.formula.api import mixedlm
import warnings
warnings.filterwarnings('ignore')

BASE_EN = 'outputs/plays_en/wordlists'
BASE_FR = 'outputs/plays_fr/wordlists'

PLAYS_EN = ['Beckett_WaitingForGodot','Beckett_Endgame','Genet_Balcony_EN','Genet_TheMaids_EN',
            'Ionesco_BaldSoprano_EN','Ionesco_Rhinoceros_EN','Becque_TheVultures','Becque_TheWomanOfParis',
            'Brieux_LaRobeRouge_EN','Brieux_WomanOnHerOwn','Dumas_Camille_EN','Dumas_TheDemiMonde_EN']
PLAYS_FR = ['Beckett_EnAttendantGodot','Beckett_FinDePartie','Genet_LeBalcon_FR','Genet_LesBonnes_FR',
            'Ionesco_LaCantriceChauve_FR','Ionesco_Rhinoceros_FR','Becque_LesCorbeaux','Becque_LaParisienne',
            'Brieux_LaRobeRouge','Brieux_LaFemmeSeule','Dumas_DameAuxCamelias','Dumas_LeDemiMonde_FR']

AUTHOR = {p: p.split('_')[0] for p in PLAYS_EN + PLAYS_FR}
GENRE  = {p: ('Absurdist' if AUTHOR[p] in ['Beckett','Genet','Ionesco'] else 'Realist')
          for p in PLAYS_EN + PLAYS_FR}

def load_convergence_data(play, lang, condition='adjacent'):
    """Load convergence by pair-feature for a single play."""
    if condition == 'adjacent':
        suffix = ''
    elif condition == 'nonadjacent':
        suffix = '_nonadjacent'
    elif condition == 'randomized':
        suffix = '_randomized'
    else:
        raise ValueError(f"Unknown condition: {condition}")

    base = BASE_EN if lang == 'en' else BASE_FR
    path = f'{base}/{play}/analysis_results/convergence_results/conv_by_pair_feature_wordlists{suffix}.csv'

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return pd.DataFrame()


def create_gender_dyad(col1, col2):
    """Undirected gender dyad (F-F, F-M, M-M): CAT does not predict a gender direction here."""
    dyad = []
    for v1, v2 in zip(col1, col2):
        if pd.isna(v1) or pd.isna(v2):
            dyad.append('unknown')
        else:
            pair = sorted([str(v1), str(v2)])
            dyad.append('-'.join(pair))
    return dyad


# Load all data for adjacent condition
print("Loading convergence data...")
all_dfs = []

for play in PLAYS_EN:
    df = load_convergence_data(play, 'en', condition='adjacent')
    if not df.empty:
        df['play'] = play
        df['author'] = AUTHOR[play]
        df['language'] = 'English'
        df['genre'] = GENRE[play]
        df['condition'] = 'adjacent'
        all_dfs.append(df)

for play in PLAYS_FR:
    df = load_convergence_data(play, 'fr', condition='adjacent')
    if not df.empty:
        df['play'] = play
        df['author'] = AUTHOR[play]
        df['language'] = 'French'
        df['genre'] = GENRE[play]
        df['condition'] = 'adjacent'
        all_dfs.append(df)

if not all_dfs:
    print("No data loaded!")
    exit(1)

# Combine
data = pd.concat(all_dfs, ignore_index=True)
print(f"Loaded {len(data)} observations from {len(set(data['play']))} plays")

# Gender dyad: undirected (theory does not predict a gender-order effect)
data['gender_dyad'] = create_gender_dyad(data['a_gender'], data['b_gender'])

# Importance dyad: DIRECTED. minor->major and major->minor are theoretically distinct
# (the CAT status-asymmetry claim).
data['importance_dyad_dir'] = data['a_category'].astype(str) + '->' + data['b_category'].astype(str)

# Dyad identifier: same A->B pair recurs once per feature family (up to 9 rows/dyad).
# This is the dependency structure flagged in the Limitations section; we model it
# as a variance component nested within play.
data['dyad_id'] = data['play'] + '::' + data['a_speaker'].astype(str) + '->' + data['b_speaker'].astype(str)

# Keep only rows with non-null convergence
data = data[data['conv'].notna()].copy()
print(f"After removing NAs: {len(data)} observations")
print(f"Unique dyads: {data['dyad_id'].nunique()}")

# Rename family to match formula
data.rename(columns={'family': 'feature_family'}, inplace=True)

# Summary statistics
print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS")
print("="*70)
print(f"\nDV (convergence): M={data['conv'].mean():.4f}, SD={data['conv'].std():.4f}")
print(f"\nGenre:\n{data['genre'].value_counts()}")
print(f"\nLanguage:\n{data['language'].value_counts()}")
print(f"\nGender dyad:\n{data['gender_dyad'].value_counts()}")
print(f"\nImportance dyad (directed):\n{data['importance_dyad_dir'].value_counts()}")
print(f"\nFeature family:\n{data['feature_family'].value_counts()}")

# Standardize DV for standardized coefficients
data_std = data.copy()
dv_mean = data_std['conv'].mean()
dv_sd = data_std['conv'].std()
data_std['conv_std'] = (data_std['conv'] - dv_mean) / dv_sd

FULL_FORMULA = 'conv ~ C(genre) + C(language) + C(gender_dyad) + C(importance_dyad_dir) + C(feature_family)'
FULL_FORMULA_STD = 'conv_std ~ C(genre) + C(language) + C(gender_dyad) + C(importance_dyad_dir) + C(feature_family)'

# Variance component for dyad, nested within play (play is the mixedlm `groups`).
# This is what actually accounts for the "nested within dyad" dependency noted in Limitations.
VC_FORMULA = {'dyad': '0 + C(dyad_id)'}

print("\n" + "="*70)
print("PRIMARY MODEL: random intercept for play + variance component for dyad")
print("(NOTE: this model does NOT include a random intercept for author -- see")
print(" robustness check below for why author is not used as a primary grouping factor)")
print("="*70)

try:
    model = mixedlm(FULL_FORMULA, data, groups=data['play'], re_formula='1', vc_formula=VC_FORMULA)
    result = model.fit(method='powell')
    print(result.summary())

    print("\n" + "="*70)
    print("PRIMARY MODEL (Standardized scale)")
    print("="*70)
    model_std = mixedlm(FULL_FORMULA_STD, data_std, groups=data_std['play'], re_formula='1', vc_formula=VC_FORMULA)
    result_std = model_std.fit(method='powell')
    print(result_std.summary())

    # ------------------------------------------------------------------
    # Fixed effects table (raw + standardized coefficients side by side)
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("FIXED EFFECTS TABLE (Raw + Standardized)")
    print("="*70)

    fe_table = pd.DataFrame({
        'coef_raw': result.fe_params,
        'se_raw': result.bse,
        'coef_std': result_std.fe_params,
        'se_std': result_std.bse,
        't': result.tvalues,
        'p': result.pvalues,
    })
    fe_table['ci_lower'] = result.fe_params - 1.96 * result.bse
    fe_table['ci_upper'] = result.fe_params + 1.96 * result.bse
    fe_table['abs_std_coef'] = fe_table['coef_std'].abs()

    print(fe_table.to_string())
    fe_table.to_csv('outputs/mixed_effects_fixed_effects.csv')
    print("\nFixed effects saved to outputs/mixed_effects_fixed_effects.csv")

    print("""
CAVEAT on reading this table directly: each row is a single contrast against a
reference level (e.g. one feature-family level vs. adverbs). The size of one
dummy-level coefficient is NOT the same thing as "how important the whole
factor is". A factor with several small, individually non-significant levels
can still be jointly significant, and vice versa. Use the likelihood-ratio
tests below to judge the importance of a whole factor (genre, language,
gender_dyad, importance_dyad_dir, feature_family) -- not the per-level
standardized-coefficient magnitudes.
""")

    # ------------------------------------------------------------------
    # Honest "relative importance": likelihood-ratio test per factor block
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("FACTOR IMPORTANCE VIA LIKELIHOOD-RATIO TESTS (ML fit, not REML)")
    print("="*70)
    print("Each row drops one whole factor from the full model and compares fit via -2*LL.")
    print("This is the correct way to test whether a multi-level categorical factor")
    print("matters overall, rather than eyeballing individual dummy coefficients.\n")

    factor_terms = {
        'genre':               'C(genre)',
        'language':            'C(language)',
        'gender_dyad':         'C(gender_dyad)',
        'importance_dyad_dir': 'C(importance_dyad_dir)',
        'feature_family':      'C(feature_family)',
    }

    # Refit full model with ML (not REML) so log-likelihoods are comparable across
    # models that differ in fixed effects.
    full_ml = mixedlm(FULL_FORMULA, data, groups=data['play'], re_formula='1', vc_formula=VC_FORMULA).fit(
        method='powell', reml=False)

    lrt_rows = []
    for name, term in factor_terms.items():
        reduced_terms = [t for key, t in factor_terms.items() if key != name]
        reduced_formula = 'conv ~ ' + ' + '.join(reduced_terms) if reduced_terms else 'conv ~ 1'
        reduced_ml = mixedlm(reduced_formula, data, groups=data['play'], re_formula='1', vc_formula=VC_FORMULA).fit(
            method='powell', reml=False)

        # df = number of fixed-effect parameters dropped
        df_dropped = len(full_ml.fe_params) - len(reduced_ml.fe_params)
        lr_stat = 2 * (full_ml.llf - reduced_ml.llf)
        lr_stat = max(lr_stat, 0.0)
        p_lrt = stats.chi2.sf(lr_stat, df_dropped) if df_dropped > 0 else np.nan

        lrt_rows.append(dict(factor=name, df=df_dropped, chi2=lr_stat, p=p_lrt))
        print(f"  {name:22s}  df={df_dropped:2d}  chi2={lr_stat:8.2f}  p={p_lrt:.4g}")

    lrt_table = pd.DataFrame(lrt_rows).sort_values('chi2', ascending=False).reset_index(drop=True)
    lrt_table.to_csv('outputs/mixed_effects_lrt_importance.csv', index=False)
    print("\nRanked by LRT chi-square (higher = more explanatory power for that factor):")
    print(lrt_table.to_string(index=False))
    print("\nLRT importance table saved to outputs/mixed_effects_lrt_importance.csv")

    # ------------------------------------------------------------------
    # Random effects / variance components
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("RANDOM EFFECTS (VARIANCE COMPONENTS) -- primary model")
    print("="*70)
    if result.cov_re is not None and result.cov_re.shape[0] > 0:
        print(f"Random intercept SD (play): {np.sqrt(result.cov_re.iloc[0, 0]):.4f}")
    else:
        print("Random intercept SD (play): not separately estimated (see dyad Var below)")
    if result.vcomp is not None and len(result.vcomp) > 0:
        for name, val in zip(result.model.exog_vc.names, result.vcomp):
            print(f"Variance component SD ({name}): {np.sqrt(val):.4f}")
    print(f"Residual SD: {np.sqrt(result.scale):.4f}")

    print("\n" + "="*70)
    print("MODEL FIT (primary, REML)")
    print("="*70)
    print(f"Log-likelihood: {result.llf:.2f}")

    # Save model-level info so the table/summary scripts don't hardcode it
    info = {
        'n_obs': len(data),
        'n_plays': data['play'].nunique(),
        'n_dyads': data['dyad_id'].nunique(),
        'sd_play': np.sqrt(result.cov_re.iloc[0, 0]) if result.cov_re is not None and result.cov_re.shape[0] > 0 else np.nan,
        'sd_residual': np.sqrt(result.scale),
        'loglik_reml': result.llf,
    }
    if result.vcomp is not None:
        for name, val in zip(result.model.exog_vc.names, result.vcomp):
            info[f'sd_{name}'] = np.sqrt(val)
    pd.Series(info).to_csv('outputs/mixed_effects_model_info.csv', header=['value'])
    print("Model info saved to outputs/mixed_effects_model_info.csv")

except Exception as e:
    print(f"Error fitting primary model: {e}")
    import traceback
    traceback.print_exc()

# ------------------------------------------------------------------
# Robustness check: nested random effects (play within author)
#
# Author is NOT used as the primary grouping factor because genre is a
# fixed, between-author attribute (each author writes in only one genre).
# Putting author in as a random-intercept grouping factor alongside a
# fixed effect for genre risks aliasing/absorbing the genre effect into
# the author-level random variance, since genre varies only at the author
# level. Instead we report this nested model as a robustness check: if the
# genre coefficient survives being tested against between-author variance,
# that is reassuring; if it does not, the primary (play-level) estimate
# should be read cautiously.
# ------------------------------------------------------------------
print("\n" + "="*70)
print("ROBUSTNESS CHECK: nested random effects, play within author")
print("(random intercept for author; play and dyad as variance components)")
print("="*70)

try:
    nested_model = mixedlm(
        FULL_FORMULA,
        data,
        groups=data['author'],
        re_formula='1',
        vc_formula={'play': '0 + C(play)', 'dyad': '0 + C(dyad_id)'},
    )
    nested_result = nested_model.fit(method='powell')
    print(nested_result.summary())

    nested_fe = pd.DataFrame({
        'coef': nested_result.fe_params,
        'se': nested_result.bse,
        'p': nested_result.pvalues,
    })
    nested_fe.to_csv('outputs/mixed_effects_nested_robustness.csv')

    print("\nComparison of genre effect across groupings:")
    for idx in nested_fe.index:
        if 'genre' in idx:
            print(f"  Nested (author intercept, play+dyad VC): coef={nested_fe.loc[idx, 'coef']:+.4f}  p={nested_fe.loc[idx, 'p']:.4f}")
    if 'fe_table' in dir():
        for idx in fe_table.index:
            if 'genre' in idx:
                print(f"  Primary (play intercept, dyad VC):       coef={fe_table.loc[idx, 'coef_raw']:+.4f}  p={fe_table.loc[idx, 'p']:.4f}")

    print("\nNested-model results saved to outputs/mixed_effects_nested_robustness.csv")

except Exception as e:
    print(f"Error fitting nested robustness model: {e}")
    import traceback
    traceback.print_exc()
