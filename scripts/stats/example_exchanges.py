"""
Find short exchanges that illustrate accommodation, for qualitative examples in the paper.

For each play, a candidate is a run of consecutive adjacent A->B pairs (turn t replies to
t-1, t+1 to t, ...) where every turn is short enough to quote. Each reply earns, for each
family it shares with the turn before it, a weight of 1 - rate_f, where rate_f is the share
of the play's turns that contain family f. Families that are frequent anyway (prepositions,
personal pronouns) therefore count little, and a run scores high when rarer families are
picked up turn after turn. The top non-overlapping runs per play are written with the
original (punctuated) text and, for each reply, the shared families and matched words.

The score is only a way to rank candidates; the examples still need to be read and chosen by hand.

Reads   outputs/plays_{en,fr}/wordlists/<Play>/{merged_turns,pairs_A_to_B}.csv
        data/plays_dialogue/<Play>/quotation_info.csv
Writes  outputs/examples/example_exchanges.md
        outputs/examples/example_exchanges.csv   (one row per candidate run)

Usage (from repo root):
  python scripts/stats/example_exchanges.py
  python scripts/stats/example_exchanges.py --turns 6 --max-tokens 25 --top 5
  python scripts/stats/example_exchanges.py --plays Beckett_WaitingForGodot Ionesco_BaldSoprano_EN
"""
import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocessing.features_wordlists import _tokenize  # noqa: E402
from preprocessing.function_word_lists import LISTS  # noqa: E402
from scripts.stats.build_alignment_table import WORKS, FAMILIES  # noqa: E402

OUT = Path('outputs/examples')


def families_in(text, lang):
    """family -> set of matched words in the text."""
    toks = _tokenize(text)
    return {fam: {t for t in toks if t in words} for fam, words in LISTS[lang].items()
            if fam in FAMILIES and any(t in words for t in toks)}


def load_turns(play, lang):
    wl = Path(f'outputs/plays_{lang}/wordlists/{play}')
    mt = pd.read_csv(wl / 'merged_turns.csv')
    quotes = pd.read_csv(f'data/plays_dialogue/{play}/quotation_info.csv', encoding='utf-8-sig')
    qtext = dict(zip(quotes.quoteID, quotes.quoteText.astype(str)))
    mt['original'] = [' '.join(qtext.get(q, '') for q in str(ids).split(','))
                      for ids in mt.source_quoteIDs.astype(str).str.replace(r'[\[\]\' ]', '', regex=True)]
    mt['fams'] = [families_in(t, lang) for t in mt.text_merged.astype(str)]
    mt['n_tokens'] = [len(_tokenize(t)) for t in mt.text_merged.astype(str)]
    pairs = pd.read_csv(wl / 'pairs_A_to_B.csv')
    return mt.set_index('merge_id'), set(zip(pairs.i_merge_id, pairs.j_merge_id))


def candidates(play, lang, n_turns, max_tokens):
    mt, adjacent = load_turns(play, lang)
    rate = {f: sum(f in fs for fs in mt.fams) / len(mt) for f in FAMILIES}
    ids = list(mt.index)
    out = []
    for s in range(len(ids) - n_turns + 1):
        run = ids[s:s + n_turns]
        if any(mt.at[m, 'n_tokens'] > max_tokens for m in run):
            continue
        if not all((a, b) in adjacent for a, b in zip(run, run[1:])):
            continue
        score, shared = 0.0, [{}]
        for a, b in zip(run, run[1:]):
            common = {f: mt.at[b, 'fams'][f] for f in mt.at[a, 'fams'] if f in mt.at[b, 'fams']}
            score += sum(1 - rate[f] for f in common)
            shared.append(common)
        out.append({'play': play, 'start': run[0], 'run': run, 'score': score, 'shared': shared})
    return mt, out


def pick(cands, top, n_turns):
    chosen = []
    for c in sorted(cands, key=lambda c: -c['score']):
        if all(abs(c['start'] - o['start']) >= n_turns for o in chosen):
            chosen.append(c)
        if len(chosen) == top:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--turns', type=int, default=5, help='turns per exchange (default 5)')
    ap.add_argument('--max-tokens', type=int, default=20, help='max tokens per turn (default 20)')
    ap.add_argument('--top', type=int, default=3, help='exchanges per play (default 3)')
    ap.add_argument('--plays', nargs='*', help='subset of plays (default: all 24)')
    args = ap.parse_args()

    plays = [(p, 'en' if i == 0 else 'fr') for pair in WORKS.values() for i, p in enumerate(pair)]
    if args.plays:
        plays = [(p, l) for p, l in plays if p in args.plays]
    names = {k: v.replace('_', ' ') for k, v in FAMILIES.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    md = ['# Candidate accommodation examples', '',
          f'Runs of {args.turns} consecutive adjacent turns, each at most {args.max_tokens} tokens, '
          'ranked by families shared with the previous turn (weighted by 1 - play base rate). '
          'After each reply: the families it shares with the previous turn, with the words of the reply.', '']
    rows = []
    for play, lang in plays:
        mt, cands = candidates(play, lang, args.turns, args.max_tokens)
        chosen = pick(cands, args.top, args.turns)
        md += [f'## {play}', '']
        if not chosen:
            md += ['_No run matches the length limits._', '']
        for k, c in enumerate(chosen, 1):
            md += [f'**Exchange {k}** (turns {c["run"][0]}-{c["run"][-1]}, score {c["score"]:.2f})', '']
            for m, common in zip(c['run'], c['shared']):
                line = f'- **{mt.at[m, "speaker"]}**: {mt.at[m, "original"].strip()}'
                if common:
                    line += '  \n  _shared: ' + '; '.join(
                        f'{names.get(f, f)} ({", ".join(sorted(w))})' for f, w in sorted(common.items())) + '_'
                md.append(line)
            md.append('')
            rows.append({'play': play, 'language': lang, 'first_turn': c['run'][0],
                         'last_turn': c['run'][-1], 'score': round(c['score'], 3),
                         'speakers': ' / '.join(mt.at[m, 'speaker'] for m in c['run'])})
    (OUT / 'example_exchanges.md').write_text('\n'.join(md), encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUT / 'example_exchanges.csv', index=False)
    print(f'{len(rows)} exchanges -> {OUT}/example_exchanges.md')


if __name__ == '__main__':
    main()
