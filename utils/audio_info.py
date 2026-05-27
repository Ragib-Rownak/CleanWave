"""
Audio Information Utility
=========================
Handles loading audio files and extracting metadata such as
duration, sample rate, channels, and format information.
"""

import os
from typing import Tuple, Optional, Dict, Any

import numpy as np
import librosa
import soundfile as sf


def load_audio(file_path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return the waveform + sample rate.

    Uses librosa for broad format support (WAV, MP3, FLAC, OGG).
    Audio is loaded as mono float32 by default for processing consistency.

    Args:
        file_path: Absolute path to the audio file.
        sr: Target sample rate. None = preserve original sample rate.

    Returns:
        Tuple of (audio_data as np.ndarray, sample_rate as int).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    supported = {".wav", ".mp3", ".flac", ".ogg"}
    if ext not in supported:
        raise ValueError(
            f"Unsupported audio format '{ext}'. Supported: {', '.join(supported)}"
        )

    # librosa.load returns (y, sr) — mono float32 by default
    audio_data, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio_data, sample_rate


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from an audio file without loading the full waveform.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dictionary with keys: filename, duration, sample_rate, channels, format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    filename = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    # Use librosa to get duration (handles all formats)
    duration = librosa.get_duration(path=file_path)

    # Try soundfile for detailed info (works for WAV, FLAC, OGG — not MP3)
    sample_rate = 0
    channels = 0
    try:
        info = sf.info(file_path)
        sample_rate = info.samplerate
        channels = info.channels
    except RuntimeError:
        # Fallback for formats soundfile can't inspect (e.g. MP3)
        audio_data, sr = librosa.load(file_path, sr=None, mono=False, duration=0.1)
        sample_rate = sr
        channels = 1 if audio_data.ndim == 1 else audio_data.shape[0]

    return {
        "filename": filename,
        "duration": round(duration, 2),
        "sample_rate": sample_rate,
        "channels": channels,
        "format": ext.replace(".", "").upper(),
    }


def format_duration(seconds: float) -> str:
    """
    Convert seconds to a human-readable MM:SS.ms string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "02:35.40".
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"
