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
    if p < 0.001: return f"{p:.4f}"
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
    df['author'] = df['play'].map(AUTHOR)

print(f"{'Author':8s}  {'adj':>6}  {'non':>6}  {'rand':>6}  {'d_rand':>7}  {'p_rand':>9}  {'d_non':>7}  {'p_non':>9}")
print("-" * 75)

for auth in ['Beckett','Ionesco','Genet','Becque','Brieux','Dumas']:
    a = adj[adj['author']==auth]['conv'].dropna().values
    n = nonadj[nonadj['author']==auth]['conv'].dropna().values
    r = rand[rand['author']==auth]['conv'].dropna().values
    _, p_rand = stats.ttest_ind(a, r)
    _, p_non  = stats.ttest_ind(a, n)
    d_rand = cohend(a, r)
    d_non  = cohend(a, n)
    print(f"{auth:8s}  {np.mean(a):.3f}   {np.mean(n):.3f}   {np.mean(r):.3f}   {d_rand:+.3f}   {fmt_p(p_rand):>9}  {d_non:+.3f}   {fmt_p(p_non):>9}")

a = adj['conv'].dropna().values
n = nonadj['conv'].dropna().values
r = rand['conv'].dropna().values
_, p_rand = stats.ttest_ind(a, r)
_, p_non  = stats.ttest_ind(a, n)
d_rand = cohend(a, r)
d_non  = cohend(a, n)
print("-" * 75)
print(f"{'Overall':8s}  {np.mean(a):.3f}   {np.mean(n):.3f}   {np.mean(r):.3f}   {d_rand:+.3f}   {fmt_p(p_rand):>9}  {d_non:+.3f}   {fmt_p(p_non):>9}")
