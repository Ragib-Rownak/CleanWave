"""
Format Conversion Utility
=========================
Converts audio between WAV, MP3, FLAC, and OGG formats using pydub (ffmpeg backend).
Also provides a direct soundfile-based export for lossless formats.
"""

import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from utils.helpers import generate_output_path, ensure_directory


# Mapping of format names to pydub export parameters
FORMAT_CONFIG = {
    "wav": {"format": "wav", "codec": None, "extension": ".wav"},
    "mp3": {"format": "mp3", "codec": "libmp3lame", "extension": ".mp3"},
    "flac": {"format": "flac", "codec": None, "extension": ".flac"},
    "ogg": {"format": "ogg", "codec": "libvorbis", "extension": ".ogg"},
    "opus": {"format": "opus", "codec": "libopus", "extension": ".opus"},
}

SUPPORTED_FORMATS = list(FORMAT_CONFIG.keys())


def export_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    output_format: str = "wav",
    output_dir: Optional[str] = None,
    filename_prefix: str = "processed",
) -> str:
    """
    Export audio data to a file in the requested format.

    For WAV and FLAC, uses soundfile for direct lossless export.
    For MP3 and OGG, writes a temporary WAV first, then converts via pydub/ffmpeg.

    Args:
        audio_data: 1D numpy array (float32, mono).
        sample_rate: Sample rate in Hz.
        output_format: Target format — one of 'wav', 'mp3', 'flac', 'ogg'.
        output_dir: Directory to write the output file. Defaults to system temp.
        filename_prefix: Prefix for the output filename.

    Returns:
        Absolute path to the exported file.

    Raises:
        ValueError: If the format is unsupported.
        RuntimeError: If ffmpeg is not available (for MP3/OGG).
    """
    output_format = output_format.lower().strip()
    if output_format not in FORMAT_CONFIG:
        raise ValueError(
            f"Unsupported format '{output_format}'. "
            f"Choose from: {', '.join(SUPPORTED_FORMATS)}"
        )

    config = FORMAT_CONFIG[output_format]

    # Determine output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="cleanwave_")
    ensure_directory(output_dir)

    output_path = generate_output_path(output_dir, filename_prefix, config["extension"])

    # --- Direct export for lossless formats ---
    if output_format in ("wav", "flac"):
        subtype = "PCM_16" if output_format == "wav" else "PCM_16"
        sf.write(output_path, audio_data, sample_rate, subtype=subtype)
        return output_path

    # --- FFmpeg-based export for lossy formats (MP3, OGG) ---
    # Step 1: Write a temporary WAV
    tmp_wav = os.path.join(output_dir, "_temp_intermediate.wav")
    sf.write(tmp_wav, audio_data, sample_rate, subtype="PCM_16")

    try:
        # Step 2: Convert with pydub
        segment = AudioSegment.from_wav(tmp_wav)
        export_kwargs = {"format": config["format"]}
        if config["codec"]:
            export_kwargs["codec"] = config["codec"]
        # Preserve sample rate
        export_kwargs["parameters"] = ["-ar", str(sample_rate)]

        segment.export(output_path, **export_kwargs)
    except Exception as e:
        # Clean up temp file before raising
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        raise RuntimeError(
            f"Failed to export as {output_format.upper()}. "
            f"Ensure ffmpeg is installed. Error: {e}"
        ) from e
    finally:
        # Always clean up intermediate WAV
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

    return output_path


def check_ffmpeg_available() -> bool:
    """
    Check whether ffmpeg is accessible on the system PATH or via imageio-ffmpeg.

    Returns:
        True if ffmpeg is available, False otherwise.
    """
    import shutil
    if shutil.which("ffmpeg") is not None:
        return True
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe and os.path.exists(ffmpeg_exe):
            # Configure pydub to use this converter
            from pydub import AudioSegment
            AudioSegment.converter = ffmpeg_exe
            return True
    except (ImportError, Exception):
        pass
    return False


# Auto-configure converter on import
check_ffmpeg_available()

