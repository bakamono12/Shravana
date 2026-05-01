"""Extract and normalize audio from video or audio files to 16kHz mono WAV."""
import subprocess
from pathlib import Path


ACCEPTED_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
ACCEPTED_AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}


def _ffmpeg_to_wav(input_path: str, output_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-vn",           # drop video
            "-ar", "16000",  # 16kHz
            "-ac", "1",      # mono
            "-acodec", "pcm_s16le",
            output_path,
        ],
        check=True,
        capture_output=True,
    )


def extract(video_path: str, out_wav: str) -> str:
    """Strip audio from video, normalize to 16kHz mono WAV. Returns out_wav path."""
    _ffmpeg_to_wav(video_path, out_wav)
    return out_wav


def normalize(audio_path: str, out_wav: str) -> str:
    """Normalize an audio file (already audio-only) to 16kHz mono WAV."""
    _ffmpeg_to_wav(audio_path, out_wav)
    return out_wav


def detect_input_type(filename: str) -> str:
    """Returns 'video' or 'audio'."""
    suffix = Path(filename).suffix.lower()
    if suffix in ACCEPTED_VIDEO:
        return "video"
    if suffix in ACCEPTED_AUDIO:
        return "audio"
    raise ValueError(f"Unsupported file type: {suffix}")


ACCEPTED_EXTENSIONS = ACCEPTED_VIDEO | ACCEPTED_AUDIO
