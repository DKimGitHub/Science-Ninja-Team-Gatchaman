# Battle of the Planets dialogue extractor

Turns an episode's English audio into a dialogue-only script using three engines
(Whisper large-v3, Parakeet, Canary) that vote on every word. Disagreements are flagged
with timestamps in `review.md` so only those clips need a human listen.

Stages: `prepare` (ffmpeg, Demucs, VAD) -> `engines` (3 models) -> `reconcile` (vote,
name snapping) -> `run` (writes `script.md`, `review.md`, `raw.json`).
Stage 4 labels speakers by voice (ECAPA embeddings on the original mix, clustered), snaps turn
changes to sentence ends, and writes `VOICE_A:`, `VOICE_B:`... It tells voices apart; it does not
name characters. Use `--speakers N` for the number of voices and `--reuse` to skip re-transcribing.

Needs network access to huggingface.co for model weights. A GPU makes it much faster.

    pip install -r requirements.txt
    python -m pipeline.run episode.mp4 out --device cuda
    python -m pytest tests
