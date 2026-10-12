"""Usage: python -m pipeline.run episode.mp4 out_dir [--device cuda] [--speakers 6] [--no-demucs]

Writes out_dir/script.md (dialogue only, one line per speaker turn) and out_dir/review.md
(flagged words with timestamps, so you only listen to those clips).

Transcription runs on Demucs-cleaned vocals. Speaker voices are matched on the ORIGINAL mix,
because Demucs smears voices together and hides speaker changes.
"""
import argparse
import json
from pathlib import Path

from . import prepare, speakers as sp
from .engines import build_engines, load_glossary
from .reconcile import norm, vote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--speakers", type=int, default=6, help="number of distinct voices in the scene")
    ap.add_argument("--no-demucs", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse out_dir/raw.json instead of re-transcribing")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    glossary = load_glossary()

    mix = prepare.to_wav(Path(a.episode), out / "full.wav")
    asr_wav = mix
    if not a.no_demucs:
        cached = out / "full_vocals16k.wav"
        asr_wav = cached if cached.exists() else prepare.separate_vocals(mix, out)
    cache = json.loads((out / "raw.json").read_text()) if a.reuse and (out / "raw.json").exists() else None
    chunks = [(r["start"], r["end"]) for r in cache] if cache else prepare.chunk_speech(asr_wav)
    engines = None if cache else build_engines(a.device)

    raw, chunk_words, review = [], [], []
    for i, (s, e) in enumerate(chunks):
        if cache:
            timed = cache[i]["timed"]
        else:
            clip = prepare.cut(asr_wav, s, e, out / "chunks" / f"{i:04d}.wav")
            timed = {en.name: [(w.text, w.start + s, w.end + s) for w in en.transcribe(clip) if norm(w.text)]
                     for en in engines}
        texts = {k: [t for t, _, _ in v] for k, v in timed.items()}
        words, flags = vote(texts, glossary)
        raw.append({"i": i, "start": s, "end": e, "timed": timed})
        if not words:
            continue
        times = sp.time_words(words, timed["parakeet"], s, e)
        flagged = {k for k, *_ in flags}
        marked = [w + (" [?]" if k in flagged else "") for k, w in enumerate(words)]
        chunk_words.append((i, marked, times))
        if flags:
            review.append(f"## {s:7.1f}s - {e:7.1f}s\nResult: {' '.join(words)}\n" +
                          "".join(f"- word {k}: {why} {votes}\n" for k, why, votes in flags))

    # Stage 4: speakers, from the original mix
    wins = sp.windows(chunks)
    labels = sp.cluster(sp.embed_windows(mix, wins), n_speakers=a.speakers)
    letter = {}
    lines = []
    for i, words, times in chunk_words:
        who = [sp.label_at(t, wins, labels) for t in times]
        who = sp.smooth(who, min_run=2)
        who = sp.snap_to_sentences(words, who)
        for spk, ws in sp.group_lines(words, who):
            letter.setdefault(spk, "VOICE_" + chr(ord("A") + len(letter)))
            lines.append(f"{letter[spk]}: {' '.join(ws)}")

    (out / "script.md").write_text("\n".join(lines) + "\n")
    (out / "review.md").write_text("\n".join(review) or "No flagged words.\n")
    (out / "raw.json").write_text(json.dumps(raw, indent=1))
    print(f"{len(lines)} lines, {len(letter)} voices, {len(review)} chunks with flagged words")


if __name__ == "__main__":
    main()
