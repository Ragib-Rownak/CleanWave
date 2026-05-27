"""
CleanWave — Browser-Based Audio Cleaner & Editor
=================================================
A production-ready Gradio application for audio processing:
  • Upload audio (WAV, MP3, FLAC, OGG)
  • Reduce noise with adjustable strength
  • Trim audio with start/end sliders
  • Convert between formats
  • Preview and download processed audio

Deployable on Hugging Face Spaces (CPU tier).
"""

import os
import atexit
import tempfile

import numpy as np
import gradio as gr
import soundfile as sf

from utils.audio_info import load_audio, get_audio_info, format_duration
from utils.noise import reduce_noise
from utils.trim import trim_audio, validate_trim_range
from utils.convert import export_audio, check_ffmpeg_available, SUPPORTED_FORMATS
from utils.helpers import (
    get_temp_dir,
    get_output_dir,
    cleanup_old_files,
    cleanup_all_temp,
    ensure_directory,
    get_file_size_mb,
)


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------
MAX_FILE_MB = 500
ensure_directory(get_temp_dir())
ensure_directory(get_output_dir())

# Register cleanup on exit
atexit.register(cleanup_all_temp)

# Warn (but don't crash) if ffmpeg is missing
if not check_ffmpeg_available():
    print(
        "⚠️  WARNING: ffmpeg not found on PATH. "
        "MP3 and OGG export will fail. Install ffmpeg to enable all formats."
    )

# Load logo as base64 to embed in HTML (removes Gradio image download/fullscreen wrappers)
import base64
import shutil

try:
    with open("logo_transparent.png", "rb") as f:
        _b64 = base64.b64encode(f.read()).decode("utf-8")
        LOGO_HTML = f'<img src="data:image/png;base64,{_b64}" id="logo-img" alt="CleanWave Logo">'
except Exception:
    LOGO_HTML = '<div id="logo-img"></div>'

# ---------------------------------------------------------------------------
# State holders — we cache loaded audio to avoid redundant disk reads
# ---------------------------------------------------------------------------
_cached_audio: dict = {"data": None, "sr": None, "path": None, "info": None}


def _cache_audio(file_path: str) -> None:
    """Load and cache the uploaded audio."""
    data, sr = load_audio(file_path, sr=None)
    info = get_audio_info(file_path)
    _cached_audio.update({"data": data, "sr": sr, "path": file_path, "info": info})


def _get_cached():
    """Return cached audio data or raise if nothing is loaded."""
    if _cached_audio["data"] is None:
        raise gr.Error("No audio loaded. Please upload a file first.")
    return _cached_audio["data"], _cached_audio["sr"], _cached_audio["info"]


# ---------------------------------------------------------------------------
# Core handler functions (wired to Gradio UI)
# ---------------------------------------------------------------------------

def handle_upload(file_path: str):
    """
    Handle a new audio upload.

    Returns updated UI components:
      - upload_status: progress/status text
      - info_text: formatted metadata
      - audio_preview: playback widget for the original
      - trim sliders: reset to full duration
    """
    if file_path is None:
        return (
            "",                      # upload_progress
            "*Upload a file to see its details.*",  # info_display
            None,                    # original_preview
            gr.update(minimum=0, maximum=300, value=0, step=0.01),
            gr.update(minimum=0, maximum=300, value=300, step=0.01),
        )

    # Periodic cleanup of old temp files (> 1 hour)
    cleanup_old_files(get_temp_dir(), max_age_seconds=3600)
    cleanup_old_files(get_output_dir(), max_age_seconds=3600)

    size_mb = get_file_size_mb(file_path)
    if size_mb > MAX_FILE_MB:
        raise gr.Error(
            f"File too large ({size_mb:.1f} MB). Maximum allowed is {MAX_FILE_MB} MB."
        )

    try:
        _cache_audio(file_path)
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"Failed to load audio: {e}")

    _, sr, info = _get_cached()
    duration = info["duration"]

    info_text = (
        f"**Filename:** {info['filename']}\n"
        f"**Duration:** {format_duration(duration)} ({duration:.2f}s)\n"
        f"**Sample Rate:** {info['sample_rate']} Hz\n"
        f"**Channels:** {info['channels']}\n"
        f"**Format:** {info['format']}\n"
        f"**Size:** {get_file_size_mb(file_path):.2f} MB"
    )

    upload_status = "✅ File uploaded and processed successfully!"

    return (
        upload_status,           # upload_progress
        info_text,               # info_display
        file_path,               # original_preview
        gr.update(minimum=0, maximum=duration, value=0, step=0.01),         # trim_start
        gr.update(minimum=0, maximum=duration, value=duration, step=0.01),  # trim_end
    )


def handle_denoise(strength: float, stationary: bool):
    """
    Apply noise reduction and return a preview audio file.
    """
    data, sr, info = _get_cached()

    reduced = reduce_noise(data, sr, strength=strength, stationary=stationary)

    # Write to temp file for playback
    out_path = os.path.join(get_temp_dir(), "denoised_preview.wav")
    sf.write(out_path, reduced, sr, subtype="PCM_16")

    return out_path


def handle_trim_preview(start_time: float, end_time: float):
    """
    Trim the (denoised or original) audio and return a preview.
    """
    data, sr, info = _get_cached()
    duration = info["duration"]

    valid, msg = validate_trim_range(start_time, end_time, duration)
    if not valid:
        raise gr.Error(msg)

    trimmed = trim_audio(data, sr, start_time, end_time)

    out_path = os.path.join(get_temp_dir(), "trimmed_preview.wav")
    sf.write(out_path, trimmed, sr, subtype="PCM_16")

    trimmed_dur = len(trimmed) / sr
    return out_path, f"Trimmed duration: **{format_duration(trimmed_dur)}** ({trimmed_dur:.2f}s)"


def _make_progress_bar(percent, label="Processing..."):
    """Generate HTML for a green pip-install style progress bar."""
    bar_color = "#39ff14" if percent < 100 else "#00e676"
    return f"""
    <div style="width:100%; margin:8px 0;">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.85rem; color:#e0e6f0; font-weight:600;">
            <span>{label}</span>
            <span>{percent}%</span>
        </div>
        <div style="width:100%; background:rgba(255,255,255,0.08); border-radius:6px; overflow:hidden; height:22px; border:1px solid rgba(0,240,255,0.12);">
            <div style="width:{percent}%; height:100%; background:linear-gradient(90deg, {bar_color}, #00c896); border-radius:6px; transition:width 0.4s ease; box-shadow:0 0 12px rgba(57,255,20,0.3);"></div>
        </div>
    </div>
    """


def handle_export(
    denoise_strength: float,
    stationary: bool,
    trim_start: float,
    trim_end: float,
    output_format: str,
    apply_denoise: bool,
    apply_trim: bool,
    progress=gr.Progress(),
):
    """
    Full processing pipeline → export.

    Applies denoise and/or trim based on user toggles, then exports
    in the requested format.  Yields intermediate progress bar updates.
    """
    data, sr, info = _get_cached()
    duration = info["duration"]
    processed = data.copy()

    # -- Step 1: Noise reduction ---
    progress(0.1, desc="Starting processing...")
    yield (
        None,                                              # export_preview
        gr.DownloadButton(visible=False),                          # export_download_btn
        "",                                                # export_status
        _make_progress_bar(10, "⏳ Starting processing..."),  # progress_bar_html
    )

    if apply_denoise:
        progress(0.2, desc="Applying noise reduction...")
        yield (
            None,
            gr.DownloadButton(visible=False),
            "",
            _make_progress_bar(20, "🔇 Applying noise reduction..."),
        )
        processed = reduce_noise(processed, sr, strength=denoise_strength, stationary=stationary)
        progress(0.5, desc="Noise reduction complete")
        yield (
            None,
            gr.DownloadButton(visible=False),
            "",
            _make_progress_bar(50, "🔇 Noise reduction complete"),
        )

    # -- Step 2: Trimming ---
    if apply_trim:
        valid, msg = validate_trim_range(trim_start, trim_end, duration)
        if not valid:
            raise gr.Error(f"Trim error: {msg}")
        progress(0.6, desc="Trimming audio...")
        yield (
            None,
            gr.DownloadButton(visible=False),
            "",
            _make_progress_bar(60, "✂️ Trimming audio..."),
        )
        processed = trim_audio(processed, sr, trim_start, trim_end)

    # -- Step 3: Export ---
    progress(0.7, desc="Exporting...")
    yield (
        None,
        gr.DownloadButton(visible=False),
        "",
        _make_progress_bar(70, "💾 Exporting to file..."),
    )

    fmt = output_format.lower().strip()
    base_name = os.path.splitext(info["filename"])[0]
    try:
        out_path = export_audio(
            processed,
            sr,
            output_format=fmt,
            output_dir=get_output_dir(),
            filename_prefix=f"{base_name}_cleaned",
        )
    except RuntimeError as e:
        raise gr.Error(str(e))

    progress(0.95, desc="Finalizing...")
    yield (
        None,
        gr.DownloadButton(visible=False),
        "",
        _make_progress_bar(95, "📦 Finalizing..."),
    )

    size_mb = get_file_size_mb(out_path)
    result_dur = len(processed) / sr

    status = (
        f"✅ **Export complete!**\n\n"
        f"- **Format:** {fmt.upper()}\n"
        f"- **Duration:** {format_duration(result_dur)}\n"
        f"- **File size:** {size_mb:.2f} MB\n"
        f"- **Denoised:** {'Yes' if apply_denoise else 'No'}\n"
        f"- **Trimmed:** {'Yes' if apply_trim else 'No'}"
    )

    progress(1.0, desc="Done!")
    yield (
        out_path,                                          # export_preview
        gr.DownloadButton(label=f"⬇️  Download {os.path.basename(out_path)}", value=out_path, visible=True),  # export_download_btn
        status,                                            # export_status
        _make_progress_bar(100, "✅ Done!"),                # progress_bar_html
    )


# ---------------------------------------------------------------------------
# Custom CSS — Dark Navy + Neon theme (default dark, with light mode toggle)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ═══════════════════════════════════════════════════════════════
   DARK MODE (default CSS variables)
   ═══════════════════════════════════════════════════════════════ */
:root {
    --cw-bg-primary:    #0a0e1a;
    --cw-bg-secondary:  #101528;
    --cw-bg-card:       #141a30;
    --cw-bg-card-hover: #1a2240;
    --cw-header-bg:     rgba(20, 26, 48, 0.4);
    --cw-border:        rgba(0, 240, 255, 0.12);
    --cw-border-hover:  rgba(0, 240, 255, 0.30);
    --cw-text:          #e0e6f0;
    --cw-text-muted:    #8892a8;
    --cw-neon-cyan:     #00f0ff;
    --cw-neon-magenta:  #ff2dff;
    --cw-neon-green:    #39ff14;
    --cw-neon-orange:   #ff9f1c;
    --cw-accent-grad:   linear-gradient(135deg, #00f0ff, #7b61ff);
    --cw-export-grad:   linear-gradient(135deg, #39ff14, #00c896);
    --cw-glow-cyan:     0 0 20px rgba(0, 240, 255, 0.25);
    --cw-glow-green:    0 0 20px rgba(57, 255, 20, 0.25);
    --cw-status-border: #00f0ff;
}

/* ═══════════════════════════════════════════════════════════════
   LIGHT MODE CSS variables (overrides when .light class is active)
   ═══════════════════════════════════════════════════════════════ */
.light {
    --cw-bg-primary:    #f5f7fb;
    --cw-bg-secondary:  #e9edf5;
    --cw-bg-card:       #ffffff;
    --cw-bg-card-hover: #f1f4fa;
    --cw-header-bg:     rgba(255, 255, 255, 0.6);
    --cw-border:        rgba(100, 110, 140, 0.15);
    --cw-border-hover:  rgba(100, 110, 140, 0.30);
    --cw-text:          #1b1f38;
    --cw-text-muted:    #64748b;
    --cw-neon-cyan:     #0284c7;
    --cw-neon-magenta:  #c026d3;
    --cw-neon-green:    #16a34a;
    --cw-neon-orange:   #ea580c;
    --cw-accent-grad:   linear-gradient(135deg, #0284c7, #6366f1);
    --cw-export-grad:   linear-gradient(135deg, #16a34a, #059669);
    --cw-glow-cyan:     0 4px 14px rgba(2, 132, 199, 0.12);
    --cw-glow-green:    0 4px 14px rgba(22, 163, 74, 0.12);
    --cw-status-border: #0284c7;
}

/* ── Global Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: var(--cw-bg-primary);
}
::-webkit-scrollbar-thumb {
    background: var(--cw-border);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--cw-neon-cyan);
}

/* ── Global Container ───────────────────────────────────────── */
.gradio-container {
    max-width: 1000px !important;
    margin: 8px auto !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: var(--cw-bg-primary) !important;
    color: var(--cw-text) !important;
    transition: background 0.3s ease, color 0.3s ease !important;
}

/* ── Header & Top Bar ───────────────────────────────────────── */
#header-container {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding: 16px 0 !important;
    margin-bottom: 0 !important;
    background: none !important;
    border: none !important;
    border-radius: 0 !important;
}

#brand-column {
    justify-content: flex-start !important;
    padding-left: 0 !important;
    margin-left: -8px !important;
}

#logo-img {
    width: 180px !important;
    height: auto !important;
    object-fit: contain !important;
    filter: drop-shadow(0 0 8px rgba(0, 240, 255, 0.3)) !important;
    margin: 0 !important;
    padding: 0 !important;
    display: block !important;
}

.light #logo-img {
    filter: drop-shadow(0 2px 6px rgba(2, 132, 199, 0.15)) !important;
}

#brand-title-desc {
    margin: 0 !important;
}

#brand-title-desc h1 {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.3px !important;
    color: var(--cw-neon-cyan) !important;
    background: none !important;
    -webkit-background-clip: unset !important;
    -webkit-text-fill-color: unset !important;
    text-shadow: 0 0 12px rgba(0, 240, 255, 0.4) !important;
    filter: none !important;
}

.light #brand-title-desc h1 {
    color: var(--cw-neon-cyan) !important;
    text-shadow: 0 2px 8px rgba(2, 132, 199, 0.2) !important;
}

#brand-title-desc .tagline {
    font-size: 0.85rem !important;
    color: var(--cw-text-muted) !important;
    margin: 2px 0 0 0 !important;
    padding: 0 !important;
    font-weight: 500 !important;
    white-space: nowrap !important;
}

#toggle-column {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    min-width: 140px !important;
}

#theme-toggle-btn {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid var(--cw-border) !important;
    color: var(--cw-text) !important;
    border-radius: 30px !important;
    padding: 8px 18px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
    width: fit-content !important;
    margin-left: auto !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

#theme-toggle-btn:hover {
    background: rgba(255, 255, 255, 0.1) !important;
    border-color: var(--cw-neon-cyan) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--cw-glow-cyan) !important;
}

.light #theme-toggle-btn {
    background: rgba(0, 0, 0, 0.03) !important;
    border-color: var(--cw-border) !important;
    color: var(--cw-text) !important;
}

.light #theme-toggle-btn:hover {
    background: rgba(0, 0, 0, 0.06) !important;
    border-color: var(--cw-neon-cyan) !important;
    box-shadow: var(--cw-glow-cyan) !important;
}

/* ── Section Cards (Main Tool Containers) ───────────────────── */
.section-card {
    border: 1px solid var(--cw-border) !important;
    border-radius: 16px !important;
    padding: 32px !important;
    margin-bottom: 24px !important;
    background: var(--cw-bg-card) !important;
    backdrop-filter: blur(8px);
    transition: box-shadow 0.3s ease, border-color 0.3s ease, background 0.3s ease !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
}
.section-card:hover {
    border-color: var(--cw-neon-cyan) !important;
    box-shadow: 0 0 0 1px var(--cw-neon-cyan), var(--cw-glow-cyan) !important;
}

/* Section headings */
.section-card h3 {
    color: var(--cw-neon-cyan) !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
    letter-spacing: 0.3px;
    margin-top: 0 !important;
}

/* ── Accent Buttons ─────────────────────────────────────────── */
.accent-btn {
    background: var(--cw-accent-grad) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform 0.15s ease, box-shadow 0.25s ease !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
.accent-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--cw-glow-cyan) !important;
}

.export-btn {
    background: var(--cw-export-grad) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    transition: transform 0.15s ease, box-shadow 0.25s ease !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
.export-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--cw-glow-green) !important;
}

/* ── Status / Metadata Box ──────────────────────────────────── */
.status-box {
    border-left: 4px solid var(--cw-status-border) !important;
    padding: 12px 16px !important;
    margin-top: 8px !important;
    background: var(--cw-bg-secondary) !important;
    border-radius: 0 8px 8px 0 !important;
    color: var(--cw-text) !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    transition: all 0.3s ease !important;
}
.status-box * {
    border-left: none !important;
}

/* ── Gradio 4 Overrides for Flat Nested UI ──────────────────── */
/* Remove nested borders on generic blocks so only the section-card glows */
.gradio-container .block {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.gradio-container .block:hover {
    border-color: transparent !important;
    box-shadow: none !important;
}

/* Form input elements */
.gradio-container input[type="text"], 
.gradio-container input[type="number"], 
.gradio-container textarea, 
.gradio-container select,
.gradio-container .dropdown-container {
    background-color: var(--cw-bg-secondary) !important;
    color: var(--cw-text) !important;
    border: 1px solid var(--cw-border) !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.05) !important;
}

.gradio-container input:focus, 
.gradio-container textarea:focus, 
.gradio-container select:focus {
    border-color: var(--cw-neon-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0, 240, 255, 0.15) !important;
}

/* Text elements & Labels */
.gradio-container .block-label,
.gradio-container label,
.gradio-container .gr-label {
    color: var(--cw-text) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

.gradio-container .block-info {
    color: var(--cw-text-muted) !important;
    font-size: 0.8rem !important;
}

/* Sliders — fix timestamp visibility */
input[type="range"] {
    accent-color: var(--cw-neon-cyan) !important;
}
input[type="range"]::-webkit-slider-runnable-track {
    background: var(--cw-bg-secondary) !important;
}

/* Ensure slider number inputs / timestamps don't get clipped */
.gradio-container .range_slider,
.gradio-container .range_slider_stop,
.gradio-container .range_slider_start {
    overflow: visible !important;
}

/* Slider value input boxes */
.gradio-container .range_slider input[type="number"],
.gradio-container .slider-input-container,
.gradio-container span[data-testid="number-input"] {
    position: relative !important;
    z-index: 5 !important;
}

/* Fix Gradio slider wrapper clipping — target all possible svelte classes */
.gradio-container [class*="wrap"][class*="svelte"] {
    overflow: visible !important;
}

/* Slider container itself needs visible overflow for end labels */
.gradio-container .slider_input_container,
.gradio-container .gradio-slider,
.gradio-container [data-testid="slider"] {
    overflow: visible !important;
}

/* The slider's min/max label text (e.g., "0" and "3641.6") */
.gradio-container .min_value,
.gradio-container .max_value {
    z-index: 5 !important;
    position: relative !important;
}

/* Ensure parent row/column doesn't clip slider labels */
.section-card .gradio-row,
.section-card .gradio-column {
    overflow: visible !important;
}

/* Add spacing below slider to prevent overlapping with next element */
.gradio-container .gradio-slider,
.gradio-container [data-testid="slider"] {
    margin-bottom: 8px !important;
    padding-bottom: 4px !important;
}

/* Checkboxes */
.gradio-container label.checkbox-label,
.gradio-container .gr-check-radio-label {
    background: var(--cw-bg-secondary) !important;
    border: 1px solid var(--cw-border) !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
    transition: all 0.2s ease !important;
    display: flex !important;
    align-items: center !important;
}

.gradio-container label.checkbox-label:hover {
    border-color: var(--cw-border-hover) !important;
    background: var(--cw-bg-card-hover) !important;
}

.gradio-container input[type="checkbox"] {
    accent-color: var(--cw-neon-cyan) !important;
}

/* File Upload / Audio Component */
.gradio-container .upload-container,
.gradio-container .file-preview,
.gradio-container .file-row {
    background: var(--cw-bg-secondary) !important;
    border: 1px solid var(--cw-border) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    overflow: hidden !important;
}

/* Audio component — allow timestamps to be visible below seekbar */
.gradio-container .type-audio {
    background: var(--cw-bg-secondary) !important;
    border: 1px solid var(--cw-border) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    overflow: visible !important;
}

/* Audio player waveform and seekbar area — prevent clipping */
.gradio-container .audio-player,
.gradio-container .waveform-container,
.gradio-container [data-testid="waveform"] {
    overflow: visible !important;
}

.gradio-container .waveform-container {
    padding-bottom: 48px !important;
}

/* Audio timestamps below the seekbar need room */
.gradio-container .timestamps {
    position: relative !important;
    top: 8px !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 5 !important;
}

/* Ensure the audio component wrapper doesn't clip */
.gradio-container .component-wrapper {
    overflow: visible !important;
}

.gradio-container .upload-container:hover {
    border-color: var(--cw-neon-cyan) !important;
}

/* ── Hide download & share mini buttons on audio players ──── */
.gradio-container audio + .icon-buttons,
.gradio-container .audio-container .icon-buttons,
.gradio-container .component-wrapper .icon-button-wrapper,
.gradio-container button[aria-label="Download"],
.gradio-container button[aria-label="Share"],
.gradio-container .download-link,
.gradio-container a[download],
.gradio-container .action-btns,
.gradio-container .actions {
    display: none !important;
}

/* Target the Gradio action icons row in audio/file components */
.gradio-container .controls .action-icon,
.gradio-container .component-toolbar {
    display: none !important;
}

/* Upload status progress styling */
#upload-progress {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: var(--cw-neon-green) !important;
    padding: 8px 0 !important;
    min-height: 32px !important;
}

#upload-progress .pending {
    color: var(--cw-neon-orange) !important;
}

/* ── Export progress bar ──────────────────────────────────── */
#export-progress-bar {
    min-height: 50px !important;
    padding: 4px 0 !important;
}

/* ── Prominent download button ────────────────────────────── */
#export-download-btn {
    background: linear-gradient(135deg, #39ff14, #00c896) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 0 20px rgba(57, 255, 20, 0.25) !important;
    margin-top: 8px !important;
}
#export-download-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 30px rgba(57, 255, 20, 0.45) !important;
}

.light #export-download-btn {
    background: linear-gradient(135deg, #16a34a, #059669) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(22, 163, 74, 0.25) !important;
}

/* Footer */
#footer-text {
    text-align: center;
    opacity: 0.5;
    font-size: 0.8rem;
    padding: 24px 0 12px;
    color: var(--cw-text-muted) !important;
}
"""

THEME_TOGGLE_JS = """
function() {
    const body = document.body;
    if (body.classList.contains('light')) {
        body.classList.remove('light');
        body.classList.add('dark');
        return "🌙 Dark Mode";
    } else {
        body.classList.remove('dark');
        body.classList.add('light');
        return "☀️ Light Mode";
    }
}
"""

INIT_THEME_JS = """
function() {
    document.body.classList.remove('light');
    document.body.classList.add('dark');
}
"""


# ---------------------------------------------------------------------------
# Gradio Blocks UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""

    with gr.Blocks(
        title="CleanWave — Audio Cleaner & Editor",
    ) as app:

        # ── Header & Top bar ────────────────────────────────────
        with gr.Row(elem_id="header-container"):
            with gr.Column(scale=9, elem_id="brand-column"):
                gr.HTML(
                    f"""
                    <div id="brand-title-desc" style="display: flex; align-items: center; gap: 16px; justify-content: flex-start;">
                        {LOGO_HTML}
                        <div>
                            <h1>CleanWave</h1>
                            <p class="tagline">All audio tools in one place</p>
                        </div>
                    </div>
                    """
                )
            with gr.Column(scale=1, min_width=140, elem_id="toggle-column"):
                mode_btn = gr.Button(
                    value="🌙 Dark Mode",
                    elem_id="theme-toggle-btn",
                )

        # ── 1. Upload Section ───────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### 📁 Upload Audio")
            audio_input = gr.File(
                label="Drag & drop or click to upload",
                file_types=[".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"],
                type="filepath",
            )
            upload_progress = gr.Markdown(
                value="",
                elem_id="upload-progress",
            )
            info_display = gr.Markdown(
                value="*Upload a file to see its details.*",
                elem_classes=["status-box"],
            )

        # ── 2. Original Preview ─────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### 🔊 Original Audio Preview")
            original_preview = gr.Audio(
                label="Original",
                interactive=False,
                type="filepath",
                buttons=[],
            )

        # ── 3. Noise Reduction ──────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### 🔇 Noise Reduction")
            with gr.Row():
                denoise_strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Denoise Strength",
                    info="0 = off, 0.3–0.5 = gentle (voice), 1.0 = maximum suppression",
                )
                stationary_toggle = gr.Checkbox(
                    label="Stationary noise mode",
                    value=False,
                    info="Enable for constant hum/hiss (AC, fan). Disable for dynamic noise.",
                )
            denoise_btn = gr.Button(
                "🎤 Preview Denoised Audio",
                variant="primary",
                elem_classes=["accent-btn"],
            )
            denoise_preview = gr.Audio(
                label="Denoised Preview",
                interactive=False,
                type="filepath",
                buttons=[],
            )

        # ── 4. Trimming ─────────────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### ✂️ Trim Audio")
            with gr.Row():
                trim_start = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=0,
                    step=0.01,
                    label="Start Time (seconds)",
                )
                trim_end = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=300,
                    step=0.01,
                    label="End Time (seconds)",
                )
            trim_btn = gr.Button(
                "✂️ Preview Trimmed Audio",
                variant="primary",
                elem_classes=["accent-btn"],
            )
            trim_preview = gr.Audio(
                label="Trimmed Preview",
                interactive=False,
                type="filepath",
                buttons=[],
            )
            trim_info = gr.Markdown("")

        # ── 5. Export ────────────────────────────────────────────
        with gr.Column(elem_classes=["section-card"]):
            gr.Markdown("### 💾 Export Processed Audio")
            with gr.Row():
                output_format = gr.Dropdown(
                    choices=["WAV", "MP3", "FLAC", "OGG", "OPUS"],
                    value="MP3",
                    label="Output Format",
                )
                apply_denoise_toggle = gr.Checkbox(
                    label="Apply noise reduction",
                    value=True,
                )
                apply_trim_toggle = gr.Checkbox(
                    label="Apply trimming",
                    value=False,
                )
            export_btn = gr.Button(
                "⬇️  Export & Download",
                variant="primary",
                elem_classes=["export-btn"],
            )
            export_progress_bar = gr.HTML(
                value="",
                elem_id="export-progress-bar",
            )
            export_preview = gr.Audio(
                label="Exported Audio Preview",
                interactive=False,
                type="filepath",
                buttons=[],
            )
            export_download_btn = gr.DownloadButton(
                label="⬇️  Download Processed File",
                visible=False,
                elem_id="export-download-btn",
            )
            export_status = gr.Markdown("", elem_classes=["status-box"])

        # ── Footer ──────────────────────────────────────────────
        gr.Markdown(
            "Made with ❤️ by Ragib Rownak",
            elem_id="footer-text",
        )

        # ── Event wiring ────────────────────────────────────────

        # Mode toggle — runs client-side JS only
        mode_btn.click(
            fn=None,
            inputs=None,
            outputs=[mode_btn],
            js=THEME_TOGGLE_JS,
        )

        # Initial theme setup on page load
        app.load(
            fn=None,
            inputs=None,
            outputs=None,
            js=INIT_THEME_JS,
        )

        # Upload → show progress then populate info + preview + reset sliders
        def _show_uploading_status(file_path):
            """Show an interim uploading status while file is being processed."""
            if file_path is None:
                return ""
            return "⏳ **Uploading & processing audio file...** Please wait."

        audio_input.upload(
            fn=_show_uploading_status,
            inputs=[audio_input],
            outputs=[upload_progress],
        )

        audio_input.change(
            fn=handle_upload,
            inputs=[audio_input],
            outputs=[upload_progress, info_display, original_preview, trim_start, trim_end],
            show_progress="full",
        )

        # Denoise preview
        denoise_btn.click(
            fn=handle_denoise,
            inputs=[denoise_strength, stationary_toggle],
            outputs=[denoise_preview],
            show_progress="full",
        )

        # Trim preview
        trim_btn.click(
            fn=handle_trim_preview,
            inputs=[trim_start, trim_end],
            outputs=[trim_preview, trim_info],
            show_progress="full",
        )

        # Export with progress bar
        export_btn.click(
            fn=handle_export,
            inputs=[
                denoise_strength,
                stationary_toggle,
                trim_start,
                trim_end,
                output_format,
                apply_denoise_toggle,
                apply_trim_toggle,
            ],
            outputs=[export_preview, export_download_btn, export_status, export_progress_bar],
            show_progress="hidden",
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        footer_links=[],
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#f5f7fb",
            body_background_fill_dark="#0a0e1a",
            background_fill_primary="#ffffff",
            background_fill_primary_dark="#101528",
            background_fill_secondary="#e9edf5",
            background_fill_secondary_dark="#141a30",
            block_background_fill="transparent",
            block_background_fill_dark="transparent",
            block_border_color="transparent",
            block_border_color_dark="transparent",
            panel_background_fill="transparent",
            panel_background_fill_dark="transparent",
            block_label_text_color="#64748b",
            block_label_text_color_dark="#e0e6f0",
            block_title_text_color="#0284c7",
            block_title_text_color_dark="#00f0ff",
            body_text_color="#1b1f38",
            body_text_color_dark="#e0e6f0",
            body_text_color_subdued="#64748b",
            body_text_color_subdued_dark="#8892a8",
            border_color_primary="rgba(100, 110, 140, 0.15)",
            border_color_primary_dark="rgba(0, 240, 255, 0.15)",
            button_primary_background_fill="linear-gradient(135deg, #0284c7, #6366f1)",
            button_primary_background_fill_dark="linear-gradient(135deg, #00f0ff, #7b61ff)",
            button_primary_text_color="#ffffff",
            button_primary_text_color_dark="#ffffff",
            button_secondary_background_fill="rgba(0, 0, 0, 0.03)",
            button_secondary_background_fill_dark="rgba(255, 255, 255, 0.05)",
            button_secondary_text_color="#1b1f38",
            button_secondary_text_color_dark="#e0e6f0",
            button_secondary_border_color="rgba(100, 110, 140, 0.15)",
            button_secondary_border_color_dark="rgba(0, 240, 255, 0.15)",
            input_background_fill="#ffffff",
            input_background_fill_dark="#0f1425",
            input_border_color="rgba(100, 110, 140, 0.2)",
            input_border_color_dark="rgba(0, 240, 255, 0.15)",
        ),
    )
