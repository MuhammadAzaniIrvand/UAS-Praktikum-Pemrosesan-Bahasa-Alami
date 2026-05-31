# app/stt.py — baris paling atas, sebelum import lain

import os
import re
import time
import subprocess
from pathlib import Path

# Set LD_LIBRARY_PATH untuk whisper.cpp shared libraries
_BASE = Path(__file__).resolve().parent.parent
_LIB1 = str(_BASE / "models" / "whisper.cpp" / "build" / "src")
_LIB2 = str(_BASE / "models" / "whisper.cpp" / "build" / "ggml" / "src")
os.environ["LD_LIBRARY_PATH"] = (
    _LIB1 + ":" + _LIB2 + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)

from app.utils import get_logger, preprocess_audio, cleanup_temp, normalize_text

logger = get_logger("stt")

BASE_DIR      = Path(__file__).resolve().parent.parent
WHISPER_BIN   = BASE_DIR / "models" / "whisper.cpp" / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = BASE_DIR / "models" / "whisper.cpp" / "models" / "ggml-medium.bin"


def transcribe(audio_path: str | Path, cleanup: bool = False) -> dict:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio tidak ditemukan: {audio_path}")

    prep_path = preprocess_audio(audio_path)
    t0        = time.time()

    cmd = [
        str(WHISPER_BIN),
        "-m",  str(WHISPER_MODEL),
        "-f",  str(prep_path),
        "--language", "auto",
        "--no-timestamps",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"whisper-cli error: {result.stderr}")

    latency = round(time.time() - t0, 2)

    lines = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if (not line
                or "whisper_" in line
                or "system_info" in line
                or "timings" in line
                or line.startswith("main:")):
            continue
        lines.append(line)

    text = " ".join(lines)
    text = re.sub(r'\[[\d:,\. \->]+\]', '', text).strip()
    text = re.sub(r'\s+', ' ', text)

    logger.info(f"[{audio_path.name}] latency={latency}s | \"{text[:60]}\"")

    if cleanup:
        cleanup_temp(prep_path)

    return {
        "text":      text,
        "language":  "auto",
        "latency_s": latency,
        "audio_id":  audio_path.stem,
    }