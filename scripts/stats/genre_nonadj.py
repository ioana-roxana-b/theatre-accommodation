import pandas as pd, numpy as np, os, scipy.stats as stats

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

def load(base, play, suf=''):
    p = f'{base}/{play}/analysis_results/convergence_results/conv_by_pair_feature_wordlists{suf}.csv'
    return pd.read_csv(p, encoding='utf-8-sig') if os.path.exists(p) else pd.DataFrame()

def cohend(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2: return np.nan
    pooled = np.sqrt(((n1-1)*np.std(a,ddof=1)**2 + (n2-1)*np.std(b,ddof=1)**2) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def fmt_p(p):
    if p < 0.0001: return "<0.0001"
    return f"{p:.4f}"

all_adj, all_rand, all_nonadj = [], [], []
for play in PLAYS_EN:
    for lst, suf in [(all_adj,''), (all_rand,'_randomized'), (all_nonadj,'_nonadjacent')]:
        df = load(BASE_EN, play, suf); df['play'] = play; lst.append(df)
for play in PLAYS_FR:
    for lst, suf in [(all_adj,''), (all_rand,'_randomized'), (all_nonadj,'_nonadjacent')]:
        df = load(BASE_FR, play, suf); df['play'] = play; lst.append(df)

adj    = pd.concat([d for d in all_adj    if not d.empty], ignore_index=True)
rand   = pd.concat([d for d in all_rand   if not d.empty], ignore_index=True)
nonadj = pd.concat([d for d in all_nonadj if not d.empty], ignore_index=True)
for df in [adj, rand, nonadj]:
    df['genre'] = df['play'].map(GENRE)

print("Genre-level stats (adj vs rand and adj vs non-adj):")
for genre in ['Absurdist', 'Realist']:
    a = adj[adj['genre']==genre]['conv'].dropna().values
    r = rand[rand['genre']==genre]['conv'].dropna().values
    n = nonadj[nonadj['genre']==genre]['conv'].dropna().values
    _, p_rand = stats.ttest_ind(a, r)
    _, p_non  = stats.ttest_ind(a, n)
    d_rand = cohend(a, r)
    d_non  = cohend(a, n)
    print(f"  {genre}: adj={np.mean(a):.3f}, non={np.mean(n):.3f}, rand={np.mean(r):.3f}")
    print(f"    adj vs rand: d={d_rand:.3f}, p={fmt_p(p_rand)}")
    print(f"    adj vs non:  d={d_non:.3f}, p={fmt_p(p_non)}")

# Also: family-level genre effect for non-adj
print("\nFamily-level: absurdist vs realist (adj condition)")
FAMILIES = sorted(adj['family'].unique())
for fam in FAMILIES:
    a = adj[adj['genre']=='Absurdist'][adj['family']==fam]['conv'].dropna().values
    r = adj[adj['genre']=='Realist'][adj['family']==fam]['conv'].dropna().values
    if len(a) < 2 or len(r) < 2: continue
    _, p2 = stats.ttest_ind(a, r)
    d2 = cohend(a, r)
    sig = '***' if p2 < 0.0056 else ('*' if p2 < 0.05 else '')
    print(f"  {fam:25s}: abs={np.mean(a):.3f}, re={np.mean(r):.3f}, d={d2:+.3f}, p={fmt_p(p2)} {sig}")
