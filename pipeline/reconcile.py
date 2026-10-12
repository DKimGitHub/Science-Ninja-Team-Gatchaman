"""Stage 3: three-way vote. No engine is trusted alone.

Backbone = the transcript closest to the other two. Each other engine is aligned to it
word by word. Each position takes the majority word. A position with no 2-of-3 majority,
or an insertion/deletion only one engine made, is flagged for human review.
"""
import re
from collections import Counter
from difflib import SequenceMatcher, get_close_matches


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9'\-]", "", t.lower())


def snap_names(word: str, glossary) -> str:
    """Pull near-miss spellings onto glossary names (e.g. 'Zoltaar' -> 'Zoltar')."""
    n = norm(word)
    if len(n) < 4:
        return word
    # names of 4 letters or fewer (Mark, Tiny) are too easy to confuse with real words
    table = {norm(g): g for g in glossary if len(norm(g)) >= 5}
    hit = get_close_matches(n, table, n=1, cutoff=0.78)
    return table[hit[0]] if hit else word


def _sim(a, b):
    return SequenceMatcher(None, [norm(x) for x in a], [norm(x) for x in b]).ratio()


def vote(texts: dict, glossary=()):
    """texts: {engine: [word, ...]} for ONE chunk.
    Returns (words, flags) where flags is a list of (index, reason, {engine: word})."""
    # Punctuation-only tokens (Canary emits "," and "?" on their own) are not words.
    texts = {n: [w for w in ws if norm(w)] for n, ws in texts.items()}
    names = list(texts)
    if len(names) < 3:
        raise ValueError("need all three engines")
    score = {n: sum(_sim(texts[n], texts[m]) for m in names if m != n) for n in names}
    back = max(names, key=score.get)
    others = [n for n in names if n != back]
    base = texts[back]

    picks = [{back: w} for w in base]          # engine -> word at each backbone slot
    extra_inserts = []                          # words only one other engine heard
    for o in others:
        sm = SequenceMatcher(None, [norm(x) for x in base], [norm(x) for x in texts[o]])
        for op, a0, a1, b0, b1 in sm.get_opcodes():
            if op in ("equal", "replace"):
                for k in range(a1 - a0):
                    j = b0 + k
                    if j < b1:
                        picks[a0 + k][o] = texts[o][j]
                    else:
                        picks[a0 + k][o] = None
            elif op == "delete":
                for i in range(a0, a1):
                    picks[i][o] = None
            elif op == "insert":
                extra_inserts.append((a0, o, texts[o][b0:b1]))

    out, flags = [], []
    for i, p in enumerate(picks):
        votes = Counter(norm(w) for w in p.values() if w is not None)
        none_votes = sum(1 for w in p.values() if w is None)
        top, cnt = votes.most_common(1)[0] if votes else ("", 0)
        if none_votes >= 2 or (cnt < 2 and none_votes == 0 and len(votes) == 3):
            reason = "dropped by majority" if none_votes >= 2 else "all three differ"
            flags.append((len(out), reason, dict(p)))
            if none_votes >= 2:
                continue
            out.append(snap_names(p[back], glossary))
            continue
        word = next(w for w in p.values() if w is not None and norm(w) == top)
        if cnt < 2:
            flags.append((len(out), "no majority", dict(p)))
        elif len(votes) > 1 or none_votes:
            pass  # 2-of-3 agreed; minority dissent is fine
        out.append(snap_names(word, glossary))
    for pos, eng, ws in extra_inserts:
        flags.append((min(pos, len(out)), f"only {eng} heard: {' '.join(ws)}", {eng: " ".join(ws)}))
    return out, flags


def confidence(flags, n_words):
    return 1.0 if n_words == 0 else max(0.0, 1.0 - len(flags) / n_words)
