# Battle of the Planets dialogue extractor

Turns an episode's English audio into a dialogue-only script using three engines
(Whisper large-v3, Parakeet, Canary) that vote on every word. Disagreements are flagged
with timestamps in `review.md` so only those clips need a human listen.

Stages: `prepare` (ffmpeg, Demucs, VAD) -> `engines` (3 models) -> `reconcile` (vote,
name snapping) -> `run` (writes `script.md`, `review.md`, `raw.json`).
Speaker attribution (diarization) is the next stage; lines are `UNKNOWN:` until then.

Needs network access to huggingface.co for model weights. A GPU makes it much faster.

    pip install -r requirements.txt
    python -m pipeline.run episode.mp4 out --device cuda
    python -m pytest tests
