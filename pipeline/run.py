"""Usage: python -m pipeline.run episode.mp4 out_dir [--device cuda] [--no-demucs]

Writes out_dir/script.md (dialogue only) and out_dir/review.md (flagged lines with
timestamps, so you only listen to those clips).
"""
import argparse
import json
from pathlib import Path

from . import prepare
from .engines import build_engines, load_glossary
from .reconcile import confidence, vote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-demucs", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    glossary = load_glossary()

    wav = prepare.to_wav(Path(a.episode), out / "full.wav")
    if not a.no_demucs:
        wav = prepare.separate_vocals(wav, out)
    chunks = prepare.chunk_speech(wav)
    engines = build_engines(a.device)

    lines, review, raw = [], [], []
    for i, (s, e) in enumerate(chunks):
        clip = prepare.cut(wav, s, e, out / "chunks" / f"{i:04d}.wav")
        texts = {en.name: [w.text for w in en.transcribe(clip)] for en in engines}
        raw.append({"i": i, "start": s, "end": e, "texts": texts})
        words, flags = vote(texts, glossary)
        if not words:
            continue
        mark = "" if not flags else " [?]"
        # Speaker attribution is a later stage; every line starts as UNKNOWN.
        lines.append(f"UNKNOWN: {' '.join(words)}{mark}")
        if flags:
            review.append(f"## {s:7.1f}s - {e:7.1f}s  (confidence {confidence(flags, len(words)):.2f})\n"
                          f"Result: {' '.join(words)}\n" +
                          "".join(f"- word {k}: {why} {votes}\n" for k, why, votes in flags))

    (out / "script.md").write_text("\n".join(lines) + "\n")
    (out / "review.md").write_text("\n".join(review) or "No flagged lines.\n")
    (out / "raw.json").write_text(json.dumps(raw, indent=1))
    print(f"{len(lines)} lines, {len(review)} flagged for review")


if __name__ == "__main__":
    main()
