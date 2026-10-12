"""Stage 4: who is talking. Voice embeddings (ECAPA) clustered into speakers."""
import numpy as np


def _embedder():
    import torch  # noqa: F401
    from speechbrain.inference.speaker import EncoderClassifier
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/ecapa", run_opts={"device": "cpu"})


def windows(chunks, win=1.5, hop=0.5, min_len=0.5):
    """Short overlapping windows inside each speech chunk: [(start, end)]."""
    out = []
    for a, b in chunks:
        if b - a < min_len:
            continue
        if b - a <= win:
            out.append((a, b))
            continue
        t = a
        while t + win <= b:
            out.append((t, t + win))
            t += hop
        if out[-1][1] < b - 0.3:
            out.append((b - win, b))
    return out


def embed_windows(wav, wins):
    import soundfile as sf
    import torch
    data, sr = sf.read(str(wav), dtype="float32")
    emb = _embedder()
    vecs = []
    for a, b in wins:
        x = torch.from_numpy(data[int(a * sr):int(b * sr)]).unsqueeze(0)
        with torch.no_grad():
            v = emb.encode_batch(x).squeeze().numpy()
        vecs.append(v / (np.linalg.norm(v) + 1e-9))
    return np.stack(vecs)


def cluster(vecs, n_speakers=None, threshold=0.55):
    from sklearn.cluster import AgglomerativeClustering
    kw = dict(n_clusters=n_speakers) if n_speakers else dict(n_clusters=None, distance_threshold=threshold)
    return AgglomerativeClustering(metric="cosine", linkage="average", **kw).fit_predict(vecs)


def label_at(t, wins, labels):
    """Speaker label of the window whose centre is nearest to time t."""
    centres = np.array([(a + b) / 2 for a, b in wins])
    return int(labels[int(np.argmin(np.abs(centres - t)))])


def time_words(final, ref, lo, hi):
    """Give each voted word a time by aligning it to one engine's timed words.
    final: [str]; ref: [(text, start, end)]; (lo, hi) = chunk bounds, used if nothing aligns."""
    from difflib import SequenceMatcher
    from .reconcile import norm
    A = [norm(w) for w in final]
    B = [norm(r[0]) for r in ref]
    t = [None] * len(final)
    for a, b, n in SequenceMatcher(None, A, B, autojunk=False).get_matching_blocks():
        for k in range(n):
            t[a + k] = (ref[b + k][1] + ref[b + k][2]) / 2
    known = [i for i, x in enumerate(t) if x is not None]
    if not known:
        return list(np.linspace(lo, hi, len(final) + 2)[1:-1])
    idx = np.arange(len(final))
    return list(np.interp(idx, known, [t[i] for i in known]))


def smooth(seq, min_run=2):
    """Absorb speaker runs shorter than min_run words into a neighbour (kills flicker)."""
    seq = list(seq)
    changed = True
    while changed and len(set(seq)) > 1:
        changed = False
        runs, i = [], 0
        while i < len(seq):
            j = i
            while j < len(seq) and seq[j] == seq[i]:
                j += 1
            runs.append([i, j])
            i = j
        for k, (a, b) in enumerate(runs):
            if b - a < min_run and len(runs) > 1:
                nb = runs[k - 1] if k > 0 else runs[k + 1]
                seq[a:b] = [seq[nb[0]]] * (b - a)
                changed = True
                break
    return seq


def group_lines(words, speakers):
    """[(speaker, [words])] with a new line whenever the speaker changes."""
    lines = []
    for w, s in zip(words, speakers):
        if lines and lines[-1][0] == s:
            lines[-1][1].append(w)
        else:
            lines.append((s, [w]))
    return lines


def snap_to_sentences(words, speakers, max_shift=3):
    """Move each speaker change to the nearest sentence end within max_shift words.
    People change turns at the end of a sentence, but window timing is a bit blurry."""
    spk = list(speakers)
    ends = lambda w: w.replace(" [?]", "").rstrip("\"')]").endswith((".", "?", "!"))
    for i in range(1, len(spk)):
        if spk[i] == spk[i - 1] or ends(words[i - 1]):
            continue
        for d in range(1, max_shift + 1):
            for j in (i - d, i + d):
                if 0 < j < len(words) and ends(words[j - 1]):
                    lo, hi = sorted((i, j))
                    side = spk[i] if j < i else spk[i - 1]
                    spk[lo:hi] = [side] * (hi - lo)
                    break
            else:
                continue
            break
    return spk
