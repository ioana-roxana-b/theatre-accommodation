def fmt(n): return f'{n:,}'

ABSURDIST = [
    ('Beckett', r'\textit{Waiting for Godot}',            'EN', 1701, 15066,  5,  12),
    ('',        r'\textit{Endgame}',                       'EN',  789,  8252,  4,  10),
    ('',        r'\textit{En attendant Godot}',            'FR', 1635, 13980,  5,  12),
    ('',        r'\textit{Fin de partie}',                 'FR',  788,  9139,  4,  11),
    ('Genet',   r'\textit{The Balcony}',                   'EN', 1004, 22591, 18,  90),
    ('',        r'\textit{The Maids}',                     'EN',  389, 11313,  3,   6),
    ('',        r'\textit{Le Balcon}',                     'FR',  965, 23777, 18,  94),
    ('',        r'\textit{Les Bonnes}',                    'FR',  404, 10247,  3,   6),
    ('Ionesco', r'\textit{The Bald Soprano}',              'EN',  482,  7117,  6,  17),
    ('',        r'\textit{Rhinoceros}',                    'EN', 1702, 22702, 17, 121),
    ('',        r'\textit{La Cantatrice Chauve}',          'FR',  483,  8169,  6,  29),
    ('',        r"\textit{Rhinoc\'{e}ros}",                'FR', 1712, 26767, 17, 109),
]

REALIST = [
    ('Becque',  r'\textit{The Vultures}',                  'EN',  783, 28493, 15,  88),
    ('',        r'\textit{The Woman of Paris}',             'EN',  485, 13044,  5,  16),
    ('',        r'\textit{Les Corbeaux}',                   'FR', 1117, 28086, 15,  86),
    ('',        r'\textit{La Parisienne}',                  'FR',  651, 13368,  5,  14),
    ('Brieux',  r'\textit{The Red Robe}$^a$',               'EN', 1069, 21806, 23, 106),
    ('',        r'\textit{Woman on Her Own}$^b$',           'EN', 1196, 23177, 26, 128),
    ('',        r'\textit{La Robe Rouge}',                  'FR',  859, 25632, 19,  93),
    ('',        r'\textit{La Femme Seule}',                 'FR', 1271, 26925, 20,  91),
    ('Dumas',   r'\textit{Camille}$^c$',                    'EN',  865, 15574, 11,  55),
    ('',        r'\textit{The Demi-Monde}$^d$',             'EN', 1423, 24977,  9,  52),
    ('',        r"\textit{La Dame aux Cam\'{e}lias}",       'FR', 1271, 21562, 15,  94),
    ('',        r'\textit{Le Demi-Monde}',                  'FR', 1686, 30486, 10,  51),
]

NOTES = [
    r'$^a$~\textit{The Red Robe} (EN) contains three speaking characters absent from the French original: '
    r'Delorme (17~turns), Old Mme Etchepare (25~turns), and The Mother (9~turns); '
    r'in \textit{La Robe Rouge}, Delorme is only mentioned by other characters, never a speaker.',
    r'$^b$~\textit{Woman on Her Own} (EN) individualises six minor ensemble workers '
    r'(e.g.\ Girard 32~turns, Deschaume 20, Charpin 19, Berthe 12) that are not '
    r'individualised characters in \textit{La Femme Seule}.',
    r'$^c$~\textit{Camille} is a simplified English theatrical adaptation, not a direct '
    r'translation; it omits Le~Comte (36~turns), Saint-Gaudens (36~turns), '
    r"Ana\"{i}s (12), Le~Docteur (14), and Arthur~(10) present in Dumas's French original.",
    r'$^d$~\textit{Le Demi-Monde} (FR) includes Sophie (6~turns), a minor character '
    r'without a counterpart in the English adaptation.',
]


def subtot(rows, lang=None):
    if lang:
        rows = [r for r in rows if r[2] == lang]
    return (sum(r[3] for r in rows), sum(r[4] for r in rows),
            sum(r[5] for r in rows), sum(r[6] for r in rows))


def play_row(author, play, lang, turns, tokens, chars, dyads):
    parts = [author, play + ' (' + lang + ')', fmt(turns), fmt(tokens), str(chars), str(dyads)]
    return ' & '.join(parts) + ' \\\\'


lines = []
lines.append(r'\begin{table*}[htbp]')
lines.append(r'\centering')
lines.append(r'\small')
lines.append(r'\caption{Corpus summary: merged dialogue turns, tokens, speaking characters, and directed dyads per play.}')
lines.append(r'\label{tab:corpus}')
lines.append(r'\begin{tabular}{llrrrr}')
lines.append(r'\toprule')
header = r'\textbf{Author} & \textbf{Play} & \textbf{Turns} & \textbf{Tokens} & \textbf{Chars} & \textbf{Dyads} \\'
lines.append(header)
lines.append(r'\midrule')
lines.append(r'\multicolumn{6}{l}{\textit{Absurdist}} \\')
lines.append(r'\midrule')
for row in ABSURDIST:
    lines.append(play_row(*row))
at = subtot(ABSURDIST)
lines.append(r'\midrule')
sub_abs = (r'\multicolumn{2}{r}{\textit{Absurdist subtotal}} & ' +
           fmt(at[0]) + ' & ' + fmt(at[1]) + ' & ' + str(at[2]) + ' & ' + str(at[3]) + r' \\')
lines.append(sub_abs)
lines.append(r'\midrule')
lines.append(r'\multicolumn{6}{l}{\textit{Realist}} \\')
lines.append(r'\midrule')
for row in REALIST:
    lines.append(play_row(*row))
rt = subtot(REALIST)
lines.append(r'\midrule')
sub_re = (r'\multicolumn{2}{r}{\textit{Realist subtotal}} & ' +
          fmt(rt[0]) + ' & ' + fmt(rt[1]) + ' & ' + str(rt[2]) + ' & ' + str(rt[3]) + r' \\')
lines.append(sub_re)
lines.append(r'\midrule')

aen = subtot(ABSURDIST, 'EN')
aren = subtot(REALIST,   'EN')
afr  = subtot(ABSURDIST, 'FR')
rfr  = subtot(REALIST,   'FR')
en   = tuple(aen[i] + aren[i] for i in range(4))
fr   = tuple(afr[i] + rfr[i]  for i in range(4))
gt   = tuple(en[i]  + fr[i]   for i in range(4))

def bold_row(label, t):
    return (r'\multicolumn{2}{r}{\textbf{' + label + r'}} & \textbf{' +
            fmt(t[0]) + r'} & \textbf{' + fmt(t[1]) + r'} & \textbf{' +
            str(t[2]) + r'} & \textbf{' + str(t[3]) + r'} \\')

lines.append(bold_row('Total EN',    en))
lines.append(bold_row('Total FR',    fr))
lines.append(bold_row('Grand Total', gt))
lines.append(r'\bottomrule')
lines.append(r'\multicolumn{6}{p{0.97\linewidth}}{\footnotesize')
for i, note in enumerate(NOTES):
    sep = r'\\[2pt]' if i < len(NOTES) - 1 else ''
    lines.append(note + sep)
lines.append(r'} \\')
lines.append(r'\end{tabular}')
lines.append(r'\end{table*}')

print('\n'.join(lines))
