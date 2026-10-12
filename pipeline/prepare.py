"""Stage 1: episode file -> clean 16 kHz mono WAV -> speech chunks."""
import subprocess
from pathlib import Path


def to_wav(src: Path, dst: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-af", "highpass=f=80,lowpass=f=7500", str(dst)],
        check=True, capture_output=True)
    return dst


def separate_vocals(wav: Path, outdir: Path) -> Path:
    """Strip music and effects with Demucs. Needs `pip install demucs`."""
    subprocess.run(
        ["python", "-m", "demucs", "--two-stems=vocals", "-o", str(outdir), str(wav)],
        check=True)
    vocals = outdir / "htdemucs" / wav.stem / "vocals.wav"
    out = outdir / f"{wav.stem}_vocals16k.wav"
    to_wav(vocals, out)
    return out


def chunk_speech(wav: Path, pad: float = 0.25, max_len: float = 28.0):
    """Voice-activity detection. Returns [(start, end)] in seconds."""
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    audio = read_audio(str(wav), sampling_rate=16000)
    stamps = get_speech_timestamps(
        audio, load_silero_vad(), sampling_rate=16000,
        min_silence_duration_ms=400, return_seconds=True)
    chunks = []
    for s in stamps:
        a, b = max(0.0, s["start"] - pad), s["end"] + pad
        while b - a > max_len:
            chunks.append((a, a + max_len))
            a += max_len
        chunks.append((a, b))
    return chunks


def cut(wav: Path, start: float, end: float, dst: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(wav), str(dst)],
        check=True, capture_output=True)
    return dst
