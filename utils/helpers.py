"""
Helper Utilities
================
Shared helper functions for file management, temporary directory handling,
and cleanup routines used across the application.
"""

import os
import glob
import time
import uuid
import tempfile
import shutil
from typing import Optional


# Default directories relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def ensure_directory(path: str) -> None:
    """Create a directory (and parents) if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def generate_output_path(
    output_dir: str, prefix: str = "processed", extension: str = ".wav"
) -> str:
    """
    Generate a unique output file path to avoid filename collisions.

    Uses a short UUID suffix for uniqueness.

    Args:
        output_dir: Target directory.
        prefix: Filename prefix.
        extension: File extension (including the dot).

    Returns:
        Full path to the uniquely-named output file.
    """
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{unique_id}{extension}"
    return os.path.join(output_dir, filename)


def cleanup_old_files(
    directory: str,
    max_age_seconds: float = 3600,
    pattern: str = "*",
) -> int:
    """
    Remove files older than `max_age_seconds` from a directory.

    Useful for periodically cleaning temporary/output directories
    to avoid unbounded disk usage on free hosting tiers.

    Args:
        directory: Path to the directory to clean.
        max_age_seconds: Maximum file age in seconds (default: 1 hour).
        pattern: Glob pattern to match files (default: all files).

    Returns:
        Number of files removed.
    """
    if not os.path.isdir(directory):
        return 0

    removed = 0
    now = time.time()

    for filepath in glob.glob(os.path.join(directory, pattern)):
        if os.path.isfile(filepath):
            file_age = now - os.path.getmtime(filepath)
            if file_age > max_age_seconds:
                try:
                    os.remove(filepath)
                    removed += 1
                except OSError:
                    pass  # Skip files that can't be removed (permissions, etc.)

    return removed


def cleanup_all_temp() -> None:
    """Remove all files in the project temp and output directories."""
    for directory in (TEMP_DIR, OUTPUT_DIR):
        if os.path.isdir(directory):
            shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory, exist_ok=True)


def get_temp_dir() -> str:
    """Get (and create) the project temp directory."""
    ensure_directory(TEMP_DIR)
    return TEMP_DIR


def get_output_dir() -> str:
    """Get (and create) the project output directory."""
    ensure_directory(OUTPUT_DIR)
    return OUTPUT_DIR


def safe_delete(filepath: str) -> bool:
    """
    Safely delete a file, ignoring errors if it doesn't exist.

    Returns:
        True if the file was deleted, False otherwise.
    """
    try:
        if filepath and os.path.isfile(filepath):
            os.remove(filepath)
            return True
    except OSError:
        pass
    return False


def get_file_size_mb(filepath: str) -> float:
    """Get file size in megabytes."""
    if os.path.isfile(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0
