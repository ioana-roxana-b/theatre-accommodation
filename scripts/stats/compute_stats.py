import pandas as pd, numpy as np, os, scipy.stats as stats, warnings
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
PLAY_NAMES = {
    'Beckett_WaitingForGodot':  'Waiting for Godot (EN)',
    'Beckett_Endgame':          'Endgame (EN)',
    'Beckett_EnAttendantGodot': 'En Attendant Godot (FR)',
    'Beckett_FinDePartie':      'Fin de Partie (FR)',
    'Genet_Balcony_EN':         'The Balcony (EN)',
    'Genet_TheMaids_EN':        'The Maids (EN)',
    'Genet_LeBalcon_FR':        'Le Balcon (FR)',
    'Genet_LesBonnes_FR':       'Les Bonnes (FR)',
    'Ionesco_BaldSoprano_EN':   'The Bald Soprano (EN)',
    'Ionesco_Rhinoceros_EN':    'Rhinoceros (EN)',
    'Ionesco_LaCantriceChauve_FR': 'La Cantatrice Chauve (FR)',
    'Ionesco_Rhinoceros_FR':    'Rhinoceros (FR)',
    'Becque_TheVultures':       'The Vultures (EN)',
    'Becque_TheWomanOfParis':   'The Woman of Paris (EN)',
    'Becque_LesCorbeaux':       'Les Corbeaux (FR)',
    'Becque_LaParisienne':      'La Parisienne (FR)',
    'Brieux_LaRobeRouge_EN':    'The Red Robe (EN)',
    'Brieux_WomanOnHerOwn':     'Woman on Her Own (EN)',
    'Brieux_LaRobeRouge':       'La Robe Rouge (FR)',
    'Brieux_LaFemmeSeule':      'La Femme Seule (FR)',
    'Dumas_Camille_EN':         'Camille (EN)',
    'Dumas_TheDemiMonde_EN':    'The Demi-Monde (EN)',
    'Dumas_DameAuxCamelias':    'La Dame aux Camelias (FR)',
    'Dumas_LeDemiMonde_FR':     'Le Demi-Monde (FR)',
}

def load(base, play, suf=''):
    p = f'{base}/{play}/analysis_results/convergence_results/conv_by_pair_feature_wordlists{suf}.csv'
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

def cohend(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled = np.sqrt(((n1-1)*np.std(a,ddof=1)**2 + (n2-1)*np.std(b,ddof=1)**2) / (n1+n2-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def bci(a, b=None, n=5000, seed=42):  # 5,000 resamples, as stated in the paper
    rng = np.random.default_rng(seed)
    if b is None:
        diffs = [np.mean(rng.choice(a, len(a))) for _ in range(n)]
    else:
        diffs = [np.mean(rng.choice(a, len(a))) - np.mean(rng.choice(b, len(b)))
                 for _ in range(n)]
    return np.percentile(diffs, [2.5, 97.5])

# ── load all data ─────────────────────────────────────────────────────────────
all_adj, all_rand, all_nonadj = [], [], []
for play in PLAYS_EN:
    for lst, suf in [(all_adj,''), (all_rand,'_randomized'), (all_nonadj,'_nonadjacent')]:
        df = load(BASE_EN, play, suf)
        df['play'] = play; df['lang'] = 'EN'; lst.append(df)
for play in PLAYS_FR:
    for lst, suf in [(all_adj,''), (all_rand,'_randomized'), (all_nonadj,'_nonadjacent')]:
        df = load(BASE_FR, play, suf)
        df['play'] = play; df['lang'] = 'FR'; lst.append(df)

adj    = pd.concat([d for d in all_adj    if not d.empty], ignore_index=True)
rand   = pd.concat([d for d in all_rand   if not d.empty], ignore_index=True)
nonadj = pd.concat([d for d in all_nonadj if not d.empty], ignore_index=True)
for df in [adj, rand, nonadj]:
    df['author'] = df['play'].map(AUTHOR)
    df['genre']  = df['play'].map(GENRE)

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("SECTION 1: Core CAT Evidence")
print("=" * 70)

a_c = adj['conv'].dropna().values
r_c = rand['conv'].dropna().values
t, p = stats.ttest_ind(a_c, r_c)
d = cohend(a_c, r_c)
print(f"Overall: adj={np.mean(a_c):.4f}, rand={np.mean(r_c):.4f}, "
      f"t={t:.2f}, p={p:.4f}, d={d:.3f}")

for auth in ['Beckett','Ionesco','Genet','Becque','Brieux','Dumas']:
    a = adj[adj['author']==auth]['conv'].dropna().values
    r = rand[rand['author']==auth]['conv'].dropna().values
    t2, p2 = stats.ttest_ind(a, r)
    d2 = cohend(a, r)
    ci = bci(a, r)
    diff = np.mean(a) - np.mean(r)
    print(f"  {auth}: adj={np.mean(a):.4f}, rand={np.mean(r):.4f}, "
          f"diff={diff:+.3f}, CI=[{ci[0]:.3f},{ci[1]:.3f}], d={d2:.3f}, p={p2:.4f}")

na_c = nonadj['conv'].dropna().values
t3, p3 = stats.ttest_ind(a_c, na_c)
d3 = cohend(a_c, na_c)
print(f"Adj>Non-adj: t={t3:.2f}, p={p3:.4f}, d={d3:.3f}")
abs_plays = [p for p in PLAYS_EN+PLAYS_FR if GENRE[p]=='Absurdist']
re_plays  = [p for p in PLAYS_EN+PLAYS_FR if GENRE[p]=='Realist']
for genre, plays in [('Absurdist',abs_plays), ('Realist',re_plays)]:
    a2 = adj[adj['play'].isin(plays)]['conv'].dropna().values
    n2 = nonadj[nonadj['play'].isin(plays)]['conv'].dropna().values
    t4, p4 = stats.ttest_ind(a2, n2)
    d4 = cohend(a2, n2)
    print(f"  {genre} adj>nonadj: d={d4:.3f}, p={p4:.4f}")

print()
print("=" * 70)
print("SECTION 2: Genre")
print("=" * 70)

abs_c = adj[adj['genre']=='Absurdist']['conv'].dropna().values
re_c  = adj[adj['genre']=='Realist']['conv'].dropna().values
t, p = stats.ttest_ind(abs_c, re_c)
d = cohend(abs_c, re_c)
print(f"Absurdist: mean={np.mean(abs_c):.4f}, n={len(abs_c)}")
print(f"Realist:   mean={np.mean(re_c):.4f},  n={len(re_c)}")
print(f"t={t:.2f}, p={p:.4f}, d={d:.3f}")
abs_rnd = rand[rand['genre']=='Absurdist']['conv'].dropna().values
re_rnd  = rand[rand['genre']=='Realist']['conv'].dropna().values
print(f"Adj-rand absurdist d={cohend(abs_c,abs_rnd):.3f}, realist d={cohend(re_c,re_rnd):.3f}")

print("Family by genre (Bonferroni a=0.0056):")
FAMILIES = sorted(adj['family'].unique())
for fam in FAMILIES:
    a = adj[(adj['genre']=='Absurdist') & (adj['family']==fam)]['conv'].dropna().values
    r = adj[(adj['genre']=='Realist')   & (adj['family']==fam)]['conv'].dropna().values
    if len(a) < 2 or len(r) < 2:
        continue
    t2, p2 = stats.ttest_ind(a, r)
    d2 = cohend(a, r)
    sig = 'Yes ***' if p2 < 0.0056 else ('No *' if p2 < 0.05 else 'No')
    print(f"  {fam}: abs={np.mean(a):.4f}, re={np.mean(r):.4f}, d={d2:.3f}, p={p2:.3f}, {sig}")

print()
print("=" * 70)
print("SECTION 3: Language")
print("=" * 70)

en_c = adj[adj['lang']=='EN']['conv'].dropna().values
fr_c = adj[adj['lang']=='FR']['conv'].dropna().values
t3, p3 = stats.ttest_ind(en_c, fr_c)
d3 = cohend(en_c, fr_c)
print(f"EN: mean={np.mean(en_c):.4f}, n={len(en_c)}")
print(f"FR: mean={np.mean(fr_c):.4f}, n={len(fr_c)}")
print(f"t={t3:.2f}, p={p3:.4f}, d={d3:.3f}")
for auth in ['Beckett','Ionesco','Genet','Becque','Brieux','Dumas']:
    e = adj[(adj['author']==auth) & (adj['lang']=='EN')]['conv'].dropna().values
    f = adj[(adj['author']==auth) & (adj['lang']=='FR')]['conv'].dropna().values
    if len(e) < 2 or len(f) < 2:
        continue
    t4, p4 = stats.ttest_ind(e, f)
    print(f"  {auth}: EN={np.mean(e):.4f}, FR={np.mean(f):.4f}, p={p4:.3f}")

print()
print("=" * 70)
print("SECTION 4: Per-play mean conv")
print("=" * 70)

for auth in ['Beckett','Ionesco','Genet','Becque','Brieux','Dumas']:
    auth_plays = [p for p in PLAYS_EN+PLAYS_FR if AUTHOR[p]==auth]
    rows = sorted([(adj[adj['play']==p]['conv'].dropna().mean(), p) for p in auth_plays], reverse=True)
    en_m = adj[(adj['author']==auth) & (adj['lang']=='EN')]['conv'].dropna().mean()
    fr_m = adj[(adj['author']==auth) & (adj['lang']=='FR')]['conv'].dropna().mean()
    a_c2 = adj[adj['author']==auth]['conv'].dropna().values
    r_c2 = rand[rand['author']==auth]['conv'].dropna().values
    d_ar = cohend(a_c2, r_c2) if len(r_c2) > 1 else np.nan
    print(f"\n{auth}: EN={en_m:.4f}, FR={fr_m:.4f}, adj-rand_d={d_ar:.3f}")
    for m, p2 in rows:
        print(f"  {PLAY_NAMES.get(p2,p2)}: {m:.3f}")
    # strongest families
    top = adj[adj['author']==auth].groupby('family')['conv'].mean().sort_values(ascending=False).head(5)
    print(f"  Top families: {dict(top.round(3))}")

print()
print("=" * 70)
print("SECTION 5: Importance")
print("=" * 70)

for label, col in [('initiator','a_category'), ('responder','b_category')]:
    for cat in ['minor','intermediate','major']:
        vals = adj[adj[col]==cat]['conv'].dropna().values
        ci = bci(vals)
        print(f"  {label} {cat}: mean={np.mean(vals):.3f}, CI=[{ci[0]:.3f},{ci[1]:.3f}], n={len(vals)}")

print("Per-author minor->major vs major->minor:")
for auth in ['Beckett','Ionesco','Dumas','Becque','Brieux','Genet']:
    mm = adj[(adj['author']==auth) & (adj['a_category']=='minor') & (adj['b_category']=='major')]['conv'].dropna().values
    Mm = adj[(adj['author']==auth) & (adj['a_category']=='major') & (adj['b_category']=='minor')]['conv'].dropna().values
    if len(mm) < 2 or len(Mm) < 2:
        print(f"  {auth}: n too small (minor->major n={len(mm)}, major->minor n={len(Mm)})")
        continue
    ci_mm = bci(mm)
    ci_Mm = bci(Mm)
    t5, p5 = stats.ttest_ind(mm, Mm)
    print(f"  {auth}: m->M={np.mean(mm):.3f} [{ci_mm[0]:.3f},{ci_mm[1]:.3f}]  "
          f"M->m={np.mean(Mm):.3f} [{ci_Mm[0]:.3f},{ci_Mm[1]:.3f}]  p={p5:.3f}")

print()
print("=" * 70)
print("SECTION 6: Gender")
print("=" * 70)

bonf = 0.05 / 6
gmap = [('M->M','M','M'), ('F->M','F','M'), ('M->F','M','F'), ('F->F','F','F')]
for label, ag, bg in gmap:
    c = adj[(adj['a_gender']==ag) & (adj['b_gender']==bg)]['conv'].dropna().values
    print(f"  {label}: mean={np.mean(c):.4f}, n={len(c)}")

mm_v = adj[(adj['a_gender']=='M') & (adj['b_gender']=='M')]['conv'].dropna().values
ff_v = adj[(adj['a_gender']=='F') & (adj['b_gender']=='F')]['conv'].dropna().values
t6, p6 = stats.ttest_ind(mm_v, ff_v)
d6 = cohend(mm_v, ff_v)
print(f"M->M vs F->F: t={t6:.3f}, p={p6:.4f}, d={d6:.3f}, bonf_sig={'Yes' if p6<bonf else 'No'}")

print("Genre x Gender:")
for genre in ['Absurdist','Realist']:
    for label, ag, bg in gmap:
        c = adj[(adj['genre']==genre) & (adj['a_gender']==ag) & (adj['b_gender']==bg)]['conv'].dropna().values
        print(f"  {genre} {label}: {np.mean(c):.3f} (n={len(c)})")
    # M->M vs F->F within genre
    mm2 = adj[(adj['genre']==genre) & (adj['a_gender']=='M') & (adj['b_gender']=='M')]['conv'].dropna().values
    ff2 = adj[(adj['genre']==genre) & (adj['a_gender']=='F') & (adj['b_gender']=='F')]['conv'].dropna().values
    if len(mm2)>1 and len(ff2)>1:
        t7, p7 = stats.ttest_ind(mm2, ff2)
        d7 = cohend(mm2, ff2)
        print(f"  {genre} M->M vs F->F: d={d7:.3f}, p={p7:.4f}")

print()
print("=" * 70)
print("SECTION 7: Network metrics")
print("=" * 70)

for lang, plays, base in [('EN', PLAYS_EN, BASE_EN), ('FR', PLAYS_FR, BASE_FR)]:
    for play in plays:
        summary = f'{base}/{play}/analysis_results/graph_results/graph_summary.txt'
        if not os.path.exists(summary):
            continue
        # "key value" per line, written by analysis/graph_analysis.py
        row = {}
        for line in open(summary, encoding='utf-8'):
            key, _, value = line.strip().partition(' ')
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value

        def f3(key):
            v = row.get(key)
            return f'{v:.3f}' if isinstance(v, float) else '?'

        print(f"  {PLAY_NAMES.get(play,play)}: "
              f"nodes={row.get('nodes', 0):.0f}, density={f3('density')}, "
              f"recip_corr={f3('reciprocity_correlation')}, "
              f"communities={row.get('communities_detected', 0):.0f}, "
              f"pos_share={f3('pos_share')}, "
              f"asym={f3('asymmetry_index_mean')}, "
              f"cat_assort={f3('category_assortativity')}")
