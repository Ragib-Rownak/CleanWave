# 🌊 CleanWave — Browser-Based Audio Cleaner & Editor

A production-ready audio processing web application built with **Gradio**, deployable on **Hugging Face Spaces**.

Upload audio → Reduce noise → Trim → Convert format → Download.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Audio Upload** | Drag-and-drop upload for WAV, MP3, FLAC, OGG with metadata display |
| **Noise Reduction** | Adjustable spectral-gating denoiser (via `noisereduce`) with voice-quality preservation |
| **Audio Trimmer** | Start/end sliders that auto-adapt to audio duration, with preview playback |
| **Format Conversion** | Export to WAV, MP3, FLAC, OGG via `pydub` + `ffmpeg` |
| **Preview & Export** | Preview each processing stage before final export and download |
| **Dark/Light Mode** | Dark navy + neon theme by default, with a toggle button at the top |

---

## 📂 Project Structure

```
CleanWave/
├── app.py                 # Main Gradio application (UI + event handlers)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── utils/
│   ├── __init__.py
│   ├── audio_info.py      # Audio loading & metadata extraction
│   ├── noise.py           # Noise reduction (noisereduce wrapper)
│   ├── trim.py            # Audio trimming with validation
│   ├── convert.py         # Format conversion (soundfile + pydub/ffmpeg)
│   └── helpers.py         # File management, temp cleanup, unique paths
├── temp/                  # Temporary preview files (auto-cleaned)
└── outputs/               # Exported files (auto-cleaned after 1 hour)
```

---

## 🚀 Quick Start (Local)

### Prerequisites

- **Python 3.9+**
- **ffmpeg** (required for MP3 and OGG support)

#### Install ffmpeg

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg -y

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg

# Verify installation
ffmpeg -version
```

### Install & Run

```bash
# Clone or navigate to the project
cd CleanWave

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py
```

The app will start at **http://localhost:7860**.

---

## 🤗 Deploy to Hugging Face Spaces

### Option 1: Upload via Web UI

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and create a new Space.
2. Select **Gradio** as the SDK.
3. Upload all project files (maintaining the directory structure).
4. The Space will auto-detect `requirements.txt` and install dependencies.
5. ffmpeg is pre-installed on HF Spaces — no extra setup needed.

### Option 2: Git Push

```bash
# Install the HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Clone your Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/cleanwave
cd cleanwave

# Copy project files into the repo
cp -r /path/to/CleanWave/* .

# Push
git add .
git commit -m "Initial deploy"
git push
```

### Space Configuration

If needed, add a `README.md` header for HF Spaces metadata:

```yaml
---
title: CleanWave
emoji: 🌊
colorFrom: cyan
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---
```

> **Note:** ffmpeg is pre-installed on Hugging Face Spaces. No `packages.txt` is needed.

---

## 🧩 Module Guide

### `utils/audio_info.py`
- **`load_audio()`** — Loads any supported audio file via librosa (returns mono float32 + sample rate).
- **`get_audio_info()`** — Extracts metadata (duration, sample rate, channels, format) without loading the full waveform.
- **`format_duration()`** — Converts seconds → `MM:SS.ms` string.

### `utils/noise.py`
- **`reduce_noise()`** — Wraps `noisereduce.reduce_noise()` with a 0–1 strength slider that maps to `prop_decrease` and threshold parameters. Supports stationary and non-stationary modes.
- **`estimate_noise_level()`** — Estimates noise floor via short-time RMS analysis (10th percentile of frame energies).

### `utils/trim.py`
- **`trim_audio()`** — Slices audio at sample-level precision for a given time range.
- **`validate_trim_range()`** — Checks that start < end, end ≤ duration, and trimmed length ≥ minimum.

### `utils/convert.py`
- **`export_audio()`** — Exports to WAV/FLAC directly via soundfile, or to MP3/OGG via a temporary WAV → pydub/ffmpeg pipeline. Preserves sample rate.
- **`check_ffmpeg_available()`** — Checks if ffmpeg is on PATH.

### `utils/helpers.py`
- **`generate_output_path()`** — Creates collision-free filenames with UUID suffixes.
- **`cleanup_old_files()`** — Removes files older than a threshold (default: 1 hour).
- **`cleanup_all_temp()`** — Wipes temp/output directories on app exit via `atexit`.

---

## ⚙️ Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Server port | `app.py` → `app.launch()` | `7860` |
| Temp file max age | `app.py` → `handle_upload()` | 3600s (1 hour) |
| Min trim length | `utils/trim.py` → `validate_trim_range()` | 0.1s |
| Denoise strength range | `app.py` → slider config | 0.0 – 1.0 |

---

## 🔮 Future Enhancements

- Waveform visualization (interactive trim via waveform click-drag)
- Batch processing (multiple files)
- AI voice enhancement / speech separation
- Equalizer controls
- Real-time noise profile estimation display

---

## 📄 License

MIT — free for personal and commercial use.
