"""
Audio Trimming Utility
======================
Handles trimming audio to a specified time range using sample-level slicing.
"""

import numpy as np


def trim_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """
    Trim audio to the specified time range.

    Args:
        audio_data: 1D numpy array of audio samples.
        sample_rate: Sample rate in Hz.
        start_time: Start time in seconds (inclusive).
        end_time: End time in seconds (inclusive).

    Returns:
        Trimmed audio as a 1D numpy array.

    Raises:
        ValueError: If the time range is invalid.
    """
    duration = len(audio_data) / sample_rate

    # Clamp values to valid range
    start_time = max(0.0, start_time)
    end_time = min(duration, end_time)

    if start_time >= end_time:
        raise ValueError(
            f"Invalid trim range: start ({start_time:.2f}s) must be less than "
            f"end ({end_time:.2f}s)."
        )

    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Clamp to array bounds
    start_sample = max(0, start_sample)
    end_sample = min(len(audio_data), end_sample)

    return audio_data[start_sample:end_sample].copy()


def validate_trim_range(
    start_time: float,
    end_time: float,
    duration: float,
    min_length: float = 0.1,
) -> tuple[bool, str]:
    """
    Validate that a trim range is sensible.

    Args:
        start_time: Proposed start time in seconds.
        end_time: Proposed end time in seconds.
        duration: Total audio duration in seconds.
        min_length: Minimum allowed trimmed length in seconds.

    Returns:
        Tuple of (is_valid: bool, message: str).
    """
    if start_time < 0:
        return False, "Start time cannot be negative."
    if end_time > duration:
        return False, f"End time ({end_time:.2f}s) exceeds audio duration ({duration:.2f}s)."
    if start_time >= end_time:
        return False, "Start time must be before end time."
    if (end_time - start_time) < min_length:
        return False, f"Trimmed audio must be at least {min_length}s long."

    return True, "Valid trim range."
