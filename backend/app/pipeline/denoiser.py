"""
Audio denoising pipeline:
1. Demucs htdemucs → extract vocals stem (drops music/ambient noise)
2. noisereduce spectral gating → clean residual hiss from vocals
3. Silero VAD → generate speech mask (used by downstream to skip silent chunks)
Outputs clean.wav and a list of speech intervals [(start_sec, end_sec)].
"""
from __future__ import annotations
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def denoise(raw_wav: str, output_dir: str, job_id: str) -> Tuple[str, List[Tuple[float, float]]]:
    """
    Returns (clean_wav_path, speech_intervals).
    speech_intervals is a list of (start, end) pairs in seconds where speech is detected.
    """
    vocals_wav = _run_demucs(raw_wav, output_dir)
    clean_wav = _run_noisereduce(vocals_wav, str(Path(output_dir) / "clean.wav"))
    speech_intervals = _run_vad(clean_wav)
    return clean_wav, speech_intervals


def _run_demucs(raw_wav: str, output_dir: str) -> str:
    """Run demucs to extract vocals stem."""
    out = Path(output_dir) / "demucs_out"
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "--two-stems=vocals",
                "--device", "cpu",
                "--out", str(out),
                raw_wav,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"demucs exited {result.returncode}:\nSTDOUT: {result.stdout[-2000:]}\nSTDERR: {result.stderr[-2000:]}"
            )
        found = list(out.rglob("vocals.wav"))
        if not found:
            raise FileNotFoundError("demucs vocals.wav not found")
        return str(found[0])
    except Exception as e:
        logger.error(f"Demucs failed ({e}), falling back to raw audio — denoising is DISABLED for this job")
        return raw_wav


def _run_noisereduce(vocals_wav: str, output_path: str) -> str:
    """Apply spectral noise gate via noisereduce."""
    try:
        import noisereduce as nr
        import soundfile as sf
        import numpy as np

        audio, sr = sf.read(vocals_wav)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        reduced = nr.reduce_noise(y=audio.astype(np.float32), sr=sr)
        sf.write(output_path, reduced, sr)
        return output_path
    except Exception as e:
        logger.warning(f"noisereduce failed ({e}), using vocals as-is")
        import shutil
        shutil.copy(vocals_wav, output_path)
        return output_path


def _run_vad(clean_wav: str) -> List[Tuple[float, float]]:
    """Use Silero VAD to detect speech intervals. Returns [(start, end)] in seconds."""
    try:
        import torch
        import soundfile as sf
        import numpy as np

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        get_speech_timestamps = utils[0]

        # Load audio with soundfile to avoid torchaudio >= 2.9 breakage
        audio_np, sr = sf.read(clean_wav, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        wav = torch.from_numpy(audio_np)

        timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)
        return [(t["start"] / 16000.0, t["end"] / 16000.0) for t in timestamps]
    except Exception as e:
        logger.warning(f"VAD failed ({e}), treating entire file as speech")
        import soundfile as sf
        info = sf.info(clean_wav)
        return [(0.0, info.duration)]


def is_chunk_silent(
    chunk_start: float,
    chunk_end: float,
    speech_intervals: List[Tuple[float, float]],
    min_speech_ratio: float = 0.1,
) -> bool:
    """Returns True if this chunk has less than min_speech_ratio speech content."""
    if not speech_intervals:
        return False
    chunk_dur = chunk_end - chunk_start
    if chunk_dur < 2.0:
        return True
    speech_in_chunk = 0.0
    for s_start, s_end in speech_intervals:
        overlap_start = max(chunk_start, s_start)
        overlap_end = min(chunk_end, s_end)
        if overlap_end > overlap_start:
            speech_in_chunk += overlap_end - overlap_start
    return (speech_in_chunk / chunk_dur) < min_speech_ratio
