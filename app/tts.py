# app/tts.py

import os
import time
import numpy as np
import soundfile as sf
from pathlib import Path

from app.utils import get_logger, AUDIO_DIR, COQUI_DIR

logger = get_logger("tts")

# ── Path model Indonesian TTS (Wikidepia VITS) ────────────────
ID_MODEL_PATH      = COQUI_DIR / "checkpoint_1260000-inference.pth"
ID_CONFIG_PATH     = COQUI_DIR / "config.json"
ID_SPEAKERS_PATH   = COQUI_DIR / "speakers.pth"
ID_DEFAULT_SPEAKER = "JV-00027"

# ── Coqui XTTS v2 untuk EN dan AR ────────────────────────────
XTTS_MODEL   = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
XTTS_SPEAKER = "Claribel Dervla"

# ── Singleton models ──────────────────────────────────────────
_id_synth = None
_xtts     = None


# Mapping teks biasa → vocab VITS Wikidepia
_ID_CHAR_MAP = {
    'c': 'tʃ',
    'g': 'ɡ',
    'v': 'f',
    'q': 'k',
    'C': 'Tʃ',
    'G': 'ɡ',
    'V': 'f',
    'Q': 'k'
}

def get_id_synth():
    global _id_synth
    if _id_synth is None:
        from TTS.utils.synthesizer import Synthesizer
        logger.info("Loading Indonesian TTS (Wikidepia VITS)...")
        _id_synth = Synthesizer(
            tts_checkpoint=str(ID_MODEL_PATH),
            tts_config_path=str(ID_CONFIG_PATH),
            tts_speakers_file=str(ID_SPEAKERS_PATH),
            use_cuda=False,
        )
        logger.info("Indonesian TTS loaded.")
    return _id_synth


def get_xtts():
    global _xtts
    if _xtts is None:
        from TTS.api import TTS
        logger.info(f"Loading Coqui XTTS: {XTTS_MODEL}")
        _xtts = TTS(model_name=XTTS_MODEL, progress_bar=False)
        logger.info("Coqui XTTS loaded.")
    return _xtts

def _normalize_for_id_tts(text: str) -> str:
    """
    Konversi teks biasa ke karakter yang ada di vocab VITS Wikidepia.
    Menghindari warning 'Character not found in vocabulary'.
    """
    result = []
    for ch in text:
        result.append(_ID_CHAR_MAP.get(ch, ch))
    return ''.join(result)

def _map_language(dominant_language: str) -> str:
    return {"ID": "id", "EN": "en", "AR": "ar"}.get(
        dominant_language.upper(), "id"
    )


def _synthesize_id(text: str, out_path: Path) -> Path:
    synth = get_id_synth()
    wav   = synth.tts(text, speaker_name=ID_DEFAULT_SPEAKER)
    sf.write(str(out_path), np.array(wav), samplerate=22050)
    return out_path


def _synthesize_xtts(text: str, lang: str, out_path: Path) -> Path:
    tts = get_xtts()
    tts.tts_to_file(
        text=text,
        language=lang,
        speaker=XTTS_SPEAKER,
        file_path=str(out_path),
    )
    return out_path


def synthesize(response_text: str,
               dominant_language: str = "ID",
               audio_id: str = None) -> dict:
    """
    Konversi teks LLM ke audio WAV.
    - ID  → Wikidepia VITS
    - EN/AR → Coqui XTTS v2
    """
    t0       = time.time()
    lang     = _map_language(dominant_language)
    suffix   = audio_id or f"tts_{int(time.time())}"
    out_path = AUDIO_DIR / f"response_{suffix}.wav"

    try:
        logger.info(f"Synthesizing | lang={lang} | text=\"{response_text[:60]}\"")

        if lang == "id":
            _synthesize_id(response_text, out_path)
        else:
            _synthesize_xtts(response_text, lang, out_path)

        latency = round(time.time() - t0, 2)
        logger.info(f"TTS selesai | {out_path.name} | latency={latency}s")

        return {
            "output_path": out_path,
            "latency_s":   latency,
            "success":     True,
            "text":        response_text,
        }

    except Exception as e:
        latency = round(time.time() - t0, 2)
        logger.error(f"TTS gagal: {e}")
        return {
            "output_path": None,
            "latency_s":   latency,
            "success":     False,
            "text":        response_text,
        }
