"""
Audio denoising pipeline:
1. Demucs htdemucs → extract vocals stem (drops music/ambient noise)
2. noisereduce spectral gating → clean residual hiss from vocals
3. Silero VAD → generate speech mask (used by downstream to skip silent chunks)
Outputs clean.wav and a list of speech intervals [(start_sec, end_sec)].
"""
from __future__ import annotations
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def denoise(
    raw_wav: str,
    output_dir: str,
    job_id: str,
    on_progress: Optional[Callable[[str, int], None]] = None,
) -> Tuple[str, List[Tuple[float, float]]]:
    """
    Returns (clean_wav_path, speech_intervals).
    on_progress(subphase, percent) is called during demucs (subphase='demucs'),
    and at fixed points for noisereduce ('noisereduce') and vad ('vad').
    Delegates to the remote GPU worker when configured (no local progress in that path).
    """
    from app.config import settings
    if settings.REMOTE_GPU_URL:
        from app.ml.remote_client import RemoteCallError
        try:
            from app.ml.remote_proxies import RemoteDenoiser
            return RemoteDenoiser().denoise(raw_wav, output_dir, job_id)
        except RemoteCallError as exc:
            if not settings.REMOTE_GPU_FALLBACK_LOCAL:
                raise
            logger.warning(f"Remote denoise failed ({exc}), falling back to local execution")

    def _demucs_progress(pct: int) -> None:
        if on_progress:
            # Scale demucs (0-100) to 0-60 of overall denoise
            on_progress("demucs", int(pct * 0.6))

    try:
        import torch as _torch
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        _device = "cpu"

    vocals_wav = _run_demucs(raw_wav, output_dir, on_progress=_demucs_progress, device=_device)
    if on_progress:
        on_progress("noisereduce", 70)
    clean_wav = _run_noisereduce(vocals_wav, str(Path(output_dir) / "clean.wav"))
    if on_progress:
        on_progress("vad", 85)
    speech_intervals = _run_vad(clean_wav)
    if on_progress:
        on_progress("vad", 100)
    return clean_wav, speech_intervals


def _run_demucs(
    raw_wav: str,
    output_dir: str,
    on_progress: Optional[Callable[[int], None]] = None,
    device: str = "cpu",
) -> str:
    """Run demucs to extract vocals stem.

    Calls on_progress(percent: int) as demucs reports progress via stderr.
    Existing callers pass no arguments beyond output_dir; behaviour is unchanged.
    """
    out = Path(output_dir) / "demucs_out"
    out.mkdir(parents=True, exist_ok=True)
    _pct_re = re.compile(r"(\d+)%")
    stderr_buf: list[str] = []
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "demucs", "--two-stems=vocals",
             "--device", device, "--out", str(out), raw_wav],
            stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
        )
        # Demucs / tqdm uses \r for in-place progress — read in small chunks
        tail = ""
        while True:
            chunk = proc.stderr.read(256)  # type: ignore[union-attr]
            if not chunk:
                break
            tail += chunk
            parts = re.split(r"[\r\n]", tail)
            tail = parts[-1]
            for part in parts[:-1]:
                stderr_buf.append(part)
                if on_progress:
                    m = _pct_re.search(part)
                    if m:
                        on_progress(int(m.group(1)))
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"demucs exited {proc.returncode}:\n{''.join(stderr_buf[-20:])}"
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

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        model = model.to(device)
        get_speech_timestamps = utils[0]

        # Load audio with soundfile to avoid torchaudio >= 2.9 breakage
        audio_np, sr = sf.read(clean_wav, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            import librosa
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        wav = torch.from_numpy(audio_np).to(device)

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
