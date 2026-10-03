import pandas as pd, os

PLAY_GROUPS = [
    ("Absurdist", [
        ("Waiting for Godot",    "EN", "data/plays_dialogue/Beckett_WaitingForGodot",     "outputs/plays_en/wordlists/Beckett_WaitingForGodot"),
        ("En Attendant Godot",   "FR", "data/plays_dialogue/Beckett_EnAttendantGodot",    "outputs/plays_fr/wordlists/Beckett_EnAttendantGodot"),
        ("Endgame",              "EN", "data/plays_dialogue/Beckett_Endgame",             "outputs/plays_en/wordlists/Beckett_Endgame"),
        ("Fin de Partie",        "FR", "data/plays_dialogue/Beckett_FinDePartie",         "outputs/plays_fr/wordlists/Beckett_FinDePartie"),
        ("The Bald Soprano",     "EN", "data/plays_dialogue/Ionesco_BaldSoprano_EN",      "outputs/plays_en/wordlists/Ionesco_BaldSoprano_EN"),
        ("La Cantatrice Chauve", "FR", "data/plays_dialogue/Ionesco_LaCantriceChauve_FR", "outputs/plays_fr/wordlists/Ionesco_LaCantriceChauve_FR"),
        ("Rhinoceros",           "EN", "data/plays_dialogue/Ionesco_Rhinoceros_EN",       "outputs/plays_en/wordlists/Ionesco_Rhinoceros_EN"),
        ("Rhinoceros",           "FR", "data/plays_dialogue/Ionesco_Rhinoceros_FR",       "outputs/plays_fr/wordlists/Ionesco_Rhinoceros_FR"),
        ("The Balcony",          "EN", "data/plays_dialogue/Genet_Balcony_EN",            "outputs/plays_en/wordlists/Genet_Balcony_EN"),
        ("Le Balcon",            "FR", "data/plays_dialogue/Genet_LeBalcon_FR",           "outputs/plays_fr/wordlists/Genet_LeBalcon_FR"),
        ("The Maids",            "EN", "data/plays_dialogue/Genet_TheMaids_EN",           "outputs/plays_en/wordlists/Genet_TheMaids_EN"),
        ("Les Bonnes",           "FR", "data/plays_dialogue/Genet_LesBonnes_FR",          "outputs/plays_fr/wordlists/Genet_LesBonnes_FR"),
    ]),
    ("Realist", [
        ("The Vultures",         "EN", "data/plays_dialogue/Becque_TheVultures",          "outputs/plays_en/wordlists/Becque_TheVultures"),
        ("Les Corbeaux",         "FR", "data/plays_dialogue/Becque_LesCorbeaux",          "outputs/plays_fr/wordlists/Becque_LesCorbeaux"),
        ("The Red Robe",         "EN", "data/plays_dialogue/Brieux_LaRobeRouge_EN",       "outputs/plays_en/wordlists/Brieux_LaRobeRouge_EN"),
        ("La Robe Rouge",        "FR", "data/plays_dialogue/Brieux_LaRobeRouge",          "outputs/plays_fr/wordlists/Brieux_LaRobeRouge"),
        ("Woman on Her Own",     "EN", "data/plays_dialogue/Brieux_WomanOnHerOwn",        "outputs/plays_en/wordlists/Brieux_WomanOnHerOwn"),
        ("La Femme Seule",       "FR", "data/plays_dialogue/Brieux_LaFemmeSeule",         "outputs/plays_fr/wordlists/Brieux_LaFemmeSeule"),
        ("Camille",              "EN", "data/plays_dialogue/Dumas_Camille_EN",            "outputs/plays_en/wordlists/Dumas_Camille_EN"),
        ("La Dame aux Cam\\'{e}lias", "FR", "data/plays_dialogue/Dumas_DameAuxCamelias", "outputs/plays_fr/wordlists/Dumas_DameAuxCamelias"),
        ("The Demi-Monde",       "EN", "data/plays_dialogue/Dumas_TheDemiMonde_EN",       "outputs/plays_en/wordlists/Dumas_TheDemiMonde_EN"),
        ("Le Demi-Monde",        "FR", "data/plays_dialogue/Dumas_LeDemiMonde_FR",        "outputs/plays_fr/wordlists/Dumas_LeDemiMonde_FR"),
    ]),
]


def get_stats(play_dir, out_dir):
    ci_path = play_dir + "/character_info.csv"
    mt_path = out_dir + "/merged_turns.csv"
    pairs_path = out_dir + "/pairs_A_to_B.csv"
    chars = len(pd.read_csv(ci_path)) if os.path.exists(ci_path) else 0
    if os.path.exists(mt_path) and os.path.exists(pairs_path):
        mt = pd.read_csv(mt_path)
        pairs = pd.read_csv(pairs_path)
        turns = len(mt)
        tokens = int(pairs["i_text"].astype(str).str.split().str.len().sum())
        dyads = int((pairs.groupby(["A", "B"]).size() >= 3).sum())
    else:
        turns = tokens = dyads = 0
    return chars, turns, tokens, dyads


lines = []
lines.append(r"\begin{table}[h!]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\begin{tabular}{llrrrr}")
lines.append(r"\toprule")
lines.append(r"\textbf{Play} & \textbf{Lang} & \textbf{Char.} & \textbf{Turns} & \textbf{Tokens} & \textbf{Dyads} \\")
lines.append(r"\midrule")

all_plays_flat = []
for genre, plays in PLAY_GROUPS:
    lines.append(r"\multicolumn{6}{l}{\textit{" + genre + r" plays}} \\[2pt]")
    for i, (title, lang, play_dir, out_dir) in enumerate(plays):
        all_plays_flat.append((title, lang, play_dir, out_dir))
        chars, turns, tokens, dyads = get_stats(play_dir, out_dir)
        row = f"{title} & {lang} & {chars} & {turns:,} & {tokens:,} & {dyads} \\\\"
        lines.append(row)
        if i % 2 == 1 and i < len(plays) - 1:
            lines.append(r"\addlinespace[3pt]")
    lines.append(r"\addlinespace[4pt]")

lines.append(r"\midrule")

en_stats = [get_stats(p, o) for _, l, p, o in all_plays_flat if l == "EN"]
fr_stats = [get_stats(p, o) for _, l, p, o in all_plays_flat if l == "FR"]
en_t = (sum(s[0] for s in en_stats), sum(s[1] for s in en_stats), sum(s[2] for s in en_stats), sum(s[3] for s in en_stats))
fr_t = (sum(s[0] for s in fr_stats), sum(s[1] for s in fr_stats), sum(s[2] for s in fr_stats), sum(s[3] for s in fr_stats))
lines.append(f"\\textbf{{Total}} & \\textbf{{EN}} & \\textbf{{{en_t[0]}}} & \\textbf{{{en_t[1]:,}}} & \\textbf{{{en_t[2]:,}}} & \\textbf{{{en_t[3]}}} \\\\")
lines.append(f"\\textbf{{Total}} & \\textbf{{FR}} & \\textbf{{{fr_t[0]}}} & \\textbf{{{fr_t[1]:,}}} & \\textbf{{{fr_t[2]:,}}} & \\textbf{{{fr_t[3]}}} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\caption{Corpus statistics per play. Turns = merged speaker turns; Tokens = word tokens in trigger turns ($A \to B$); Dyads = speaker pairs with $\geq 3$ trigger turns.}")
lines.append(r"\label{tab:corpus}")
lines.append(r"\end{table}")

print("\n".join(lines))
