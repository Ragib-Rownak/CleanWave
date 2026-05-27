"""
Noise Reduction Utility
=======================
Provides spectral-gating noise reduction using the `noisereduce` library.
Includes adjustable strength and voice-quality preservation logic.
"""

import numpy as np
import noisereduce as nr


def reduce_noise(
    audio_data: np.ndarray,
    sample_rate: int,
    strength: float = 0.5,
    stationary: bool = False,
) -> np.ndarray:
    """
    Apply noise reduction to an audio signal.

    Uses spectral gating via noisereduce. The `strength` parameter controls
    how aggressively noise is suppressed — lower values preserve more of the
    original signal (better for voice), higher values remove more noise.

    Args:
        audio_data: 1D numpy array of audio samples (float32, mono).
        sample_rate: Sample rate of the audio in Hz.
        strength: Noise reduction intensity from 0.0 (none) to 1.0 (maximum).
                  Recommended range for voice: 0.3–0.6.
        stationary: If True, assumes noise is stationary (constant hum/hiss).
                    If False, uses non-stationary mode (adapts to changing noise).

    Returns:
        Noise-reduced audio as a 1D numpy array.
    """
    # Clamp strength to valid range
    strength = max(0.0, min(1.0, strength))

    if strength == 0.0:
        # No reduction requested — return original
        return audio_data.copy()

    # Map the 0–1 strength slider to noisereduce parameters.
    # prop_decrease: proportion of noise to remove (0 = none, 1 = all)
    # n_std_thresh_stationary: threshold for stationary noise detection
    prop_decrease = strength
    n_std_thresh = 1.5 + (1.0 - strength) * 2.0  # Lower strength → higher threshold → gentler

    reduced = nr.reduce_noise(
        y=audio_data,
        sr=sample_rate,
        prop_decrease=prop_decrease,
        n_std_thresh_stationary=n_std_thresh,
        stationary=stationary,
    )

    return reduced


def estimate_noise_level(audio_data: np.ndarray, sample_rate: int) -> float:
    """
    Estimate the noise level of an audio signal.

    Uses the RMS of the quietest 10% of short-time frames as a rough proxy.

    Args:
        audio_data: Audio waveform (1D float32 array).
        sample_rate: Sample rate in Hz.

    Returns:
        Estimated noise RMS level (float).
    """
    # Split into short frames (~25ms each)
    frame_length = int(0.025 * sample_rate)
    hop_length = frame_length // 2
    n_frames = max(1, (len(audio_data) - frame_length) // hop_length)

    rms_values = []
    for i in range(n_frames):
        start = i * hop_length
        frame = audio_data[start : start + frame_length]
        rms = np.sqrt(np.mean(frame**2))
        rms_values.append(rms)

    rms_values = np.array(rms_values)

    # Take the 10th percentile as the noise floor estimate
    noise_floor = np.percentile(rms_values, 10)
    return float(noise_floor)
