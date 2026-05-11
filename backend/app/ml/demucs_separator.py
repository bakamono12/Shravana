"""Demucs htdemucs vocal separator — strips music/noise, keeps vocals stem."""
import subprocess
import sys
from pathlib import Path


class DemucsSeparator:
    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name

    def separate_vocals(self, audio_path: str, output_dir: str) -> str:
        """
        Run demucs CLI to extract vocals stem.
        Returns path to the vocals WAV file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        subprocess.run(
            [
                sys.executable, "-m", "demucs",
                "--two-stems=vocals",
                "--name", self.model_name,
                "--device", device,
                "--out", str(out),
                audio_path,
            ],
            check=True,
            capture_output=True,
        )

        # demucs writes: out/{model_name}/{stem_name}/vocals.wav
        stem_name = Path(audio_path).stem
        vocals_path = out / self.model_name / stem_name / "vocals.wav"
        if not vocals_path.exists():
            # Some demucs versions use different directory layout
            # Try to find vocals.wav anywhere under out
            found = list(out.rglob("vocals.wav"))
            if not found:
                raise FileNotFoundError(f"demucs vocals output not found under {out}")
            vocals_path = found[0]

        return str(vocals_path)
