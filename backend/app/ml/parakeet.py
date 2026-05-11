"""NVIDIA Parakeet TDT 1.1B — English ASR with native word timestamps."""
from typing import List
from app.ml.base import BaseSTTModel, TranscriptSegment, WordTimestamp


class ParakeetModel(BaseSTTModel):
    def __init__(self, model_id: str = "nvidia/parakeet-tdt-1.1b"):
        self.model_id = model_id
        self._model = None

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr  # noqa
        import torch
        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_id)
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        self._model.eval()

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        if self._model is None:
            raise RuntimeError("ParakeetModel not loaded")
        output = self._model.transcribe([audio_path], timestamps=True)
        segments = []
        for hyp in output:
            words = []
            if hasattr(hyp, "timestamp") and hyp.timestamp:
                for w in hyp.timestamp.get("word", []):
                    words.append(WordTimestamp(
                        word=w["word"],
                        start=w["start_offset"] / 1000.0,
                        end=w["end_offset"] / 1000.0,
                    ))
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            start = words[0].start if words else 0.0
            end = words[-1].end if words else 0.0
            segments.append(TranscriptSegment(text=text, start=start, end=end, words=words, language="en"))
        return segments
