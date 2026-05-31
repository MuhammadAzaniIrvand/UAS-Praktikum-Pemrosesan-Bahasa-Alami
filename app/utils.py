# app/utils.py

import os
import re
import logging
import subprocess
import unicodedata
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Direktori (sesuai struktur folder proyek) ─────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
LOG_DIR         = BASE_DIR / "log"
DATA_DIR        = BASE_DIR / "data"
CORPUS_DIR      = DATA_DIR / "corpus"
AUDIO_DIR       = CORPUS_DIR / "audio"
TRANSCRIPT_DIR  = CORPUS_DIR / "transcripts"
MANIFEST_DIR    = DATA_DIR / "manifests"
MODELS_DIR      = BASE_DIR / "models"
WHISPER_DIR     = MODELS_DIR / "whisper.cpp"
COQUI_DIR       = BASE_DIR / "app" / "coqui_tts"

# Buat semua folder jika belum ada
for _d in [
    LOG_DIR, AUDIO_DIR, TRANSCRIPT_DIR,
    MANIFEST_DIR, MODELS_DIR, WHISPER_DIR, COQUI_DIR
]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Logger ────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        fh = logging.FileHandler(
            LOG_DIR / f"{name}.log",
            encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Audio preprocessing ───────────────────────────────────────
def preprocess_audio(audio_path: str | Path,
                     output_path: str | Path = None) -> Path:
    """
    Standarisasi audio:
    - Format  : WAV
    - Channel : mono
    - Rate    : 16000 Hz
    - Volume  : loudnorm
    Output disimpan di data/corpus/audio/ dengan prefix 'prep_'
    """
    logger     = get_logger("utils")
    audio_path = Path(audio_path)

    if output_path is None:
        output_path = AUDIO_DIR / f"prep_{audio_path.stem}.wav"
    output_path = Path(output_path)

    cmd = [
        "ffmpeg", "-y",
        "-i",  str(audio_path),
        "-ar", "16000",
        "-ac", "1",
        "-af", "loudnorm",
        "-f",  "wav",
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"ffmpeg error: {result.stderr}")
        raise RuntimeError(f"Preprocessing gagal:\n{result.stderr}")

    logger.info(f"Audio diproses: {audio_path.name} → {output_path.name}")
    return output_path


# ── Cleanup file prep sementara ───────────────────────────────
def cleanup_temp(file_path: str | Path) -> None:
    """Hapus file preprocessing sementara (prefix 'prep_')."""
    try:
        p = Path(file_path)
        if p.exists() and p.stem.startswith("prep_"):
            p.unlink()
            get_logger("utils").info(f"File temp dihapus: {p.name}")
    except Exception as e:
        get_logger("utils").warning(f"Gagal hapus {file_path}: {e}")


# ── Normalize text (wrapper — logika ada di text_processor) ───
def normalize_text(text: str) -> str:
    """
    Dipakai oleh stt.py untuk evaluasi WER/CER.
    Logika lengkap ada di text_processor.normalize().
    """
    from app.text_processor import normalize
    return normalize(text)