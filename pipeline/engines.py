"""Stage 2: the three engines behind one interface.

Every engine returns a list of Word(text, start, end, conf) for one audio file.
Weights download from Hugging Face on first use.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float
    conf: float = 1.0


def load_glossary(path="glossary.txt"):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


class WhisperEngine:
    name = "whisper"

    def __init__(self, glossary, device="cpu", model="large-v3"):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device=device,
                              compute_type="int8" if device == "cpu" else "float16")
        self.prompt = "Names: " + ", ".join(glossary) + "."

    def transcribe(self, wav):
        segs, _ = self.m.transcribe(
            str(wav), language="en", word_timestamps=True, beam_size=5,
            initial_prompt=self.prompt, condition_on_previous_text=False,
            vad_filter=False, no_speech_threshold=0.5)
        return [Word(w.word.strip(), w.start, w.end, w.probability)
                for s in segs for w in (s.words or [])]


class ParakeetEngine:
    name = "parakeet"

    def __init__(self, glossary=None, device="cpu"):
        import nemo.collections.asr as nemo_asr
        self.m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
        self.m.to(device)

    def transcribe(self, wav):
        out = self.m.transcribe([str(wav)], timestamps=True)[0]
        return [Word(w["word"], w["start"], w["end"]) for w in out.timestamp["word"]]


class CanaryEngine:
    name = "canary"

    def __init__(self, glossary=None, device="cpu"):
        from nemo.collections.asr.models import EncDecMultiTaskModel
        self.m = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-flash")
        self.m.to(device)

    def transcribe(self, wav):
        out = self.m.transcribe([str(wav)], source_lang="en", target_lang="en",
                                pnc="yes", timestamps=True)[0]
        return [Word(w["word"], w["start"], w["end"]) for w in out.timestamp["word"]]


def build_engines(device="cpu"):
    g = load_glossary()
    return [WhisperEngine(g, device), ParakeetEngine(g, device), CanaryEngine(g, device)]
