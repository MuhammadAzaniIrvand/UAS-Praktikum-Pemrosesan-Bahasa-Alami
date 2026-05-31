# app/text_processor.py

import re
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from app.utils import get_logger, MANIFEST_DIR

logger = get_logger("text_processor")


# ── Data structures ───────────────────────────────────────────

@dataclass
class TextSegment:
    text: str
    language: str       # "ID", "EN", "AR", "UNKNOWN"
    confidence: float = 1.0

    def __repr__(self):
        return f"[{self.language}] \"{self.text}\""


@dataclass
class ProcessedText:
    raw: str
    normalized: str
    segments: list[TextSegment] = field(default_factory=list)
    dominant_language: str = "ID"
    languages_detected: list[str] = field(default_factory=list)
    for_llm: str = ""

    def __repr__(self):
        return (
            f"ProcessedText(\n"
            f"  normalized   = {self.normalized!r}\n"
            f"  dominant     = {self.dominant_language}\n"
            f"  languages    = {self.languages_detected}\n"
            f"  segments     = {self.segments}\n"
            f"  for_llm      = {self.for_llm!r}\n"
            f")"
        )


# ── Kamus normalisasi ejaan ───────────────────────────────────

_NORM_MAP = {
    # Variasi ejaan Arab-Latin
    "insyaallah":    "insyaAllah",
    "inshallah":     "insyaAllah",
    "insha allah":   "insyaAllah",
    "alhamdulilah":  "alhamdulillah",
    "alhamdullilah": "alhamdulillah",
    "masya allah":   "masyaAllah",
    "masyaallah":    "masyaAllah",
    "subhanallah":   "subhanAllah",
    "subhanaallah":  "subhanAllah",
    # Kata tidak baku Bahasa Indonesia
    "gak":           "tidak",
    "ga":            "tidak",
    "nggak":         "tidak",
    "udah":          "sudah",
    "udh":           "sudah",
    # Nama tempat — kapitalisasi konsisten
    "jeddah":        "Jeddah",
    "makkah":        "Makkah",
    "madinah":       "Madinah",
    "saudi":         "Saudi",
}


# ── Kamus CS → Bahasa Indonesia (normalize) ───────────────────

_CS_TO_ID_MAP = {
    # ── EN → ID : Travel & Transport ─────────────────────────
    "book": "pesan", "booking":"pemesanan", "flight": "penerbangan", "schedule": "jadwal",
    "transport": "transportasi", "arrange": "mengatur", "trip": "perjalanan", "travel": "perjalanan",
    "visit": "kunjungan", "include": "termasuk", "simple": "sederhana", "direct": "langsung",
    "tomorrow": "besok", "next": "depan", "week": "minggu", "today": "hari ini", "after": "setelah",
    "from": "dari", "best": "terbaik",

    # ── EN → ID : Dokumen & Admin ─────────────────────────────
    "visa":"visa", "apply":"mengajukan", "step":"langkah", "guide":"panduan", "checklist":"daftar persiapan",
    "prepare":"mempersiapkan", "document":"dokumen", "online":"daring", "email":"surel", "attachment":"lampiran", 
    "upload":"unggah", "check":"periksa", "review":"tinjau", "submit":"kirim", "assignment":"tugas",
    "deadline":"batas waktu","presentation": "presentasi","translate": "terjemahkan",

    # ── EN → ID : Hotel & Akomodasi ───────────────────────────
    "hotel":"hotel", "budget":"anggaran",

    # ── EN → ID : Umum ────────────────────────────────────────
    "help": "bantu", "explain": "jelaskan", "tips": "saran", "overwhelmed": "kewalahan", "beginner": "pemula",
    "meeting": "rapat", "session": "sesi", "how": "bagaimana", "can": "bisa", "you": "kamu", "please": "tolong",
    "with": "dengan", "for": "untuk", "and": "dan", "by" :"demi",
    
    # ── AR transliterasi → ID ─────────────────────────────────
    "uridu": "saya ingin", "mumkin": "bisakah", "ahyanan": "kadang-kadang", "min": "dari", "ila": "ke",
    "ghadan": "besok", "hal": "apakah", "afdhal": "terbaik", "rihlah": "perjalanan",
    "mubasyarah": "langsung", "ya akhi": "wahai saudaraku", "usbu": "minggu",
}


# ── Kamus kata kunci per bahasa (language tagging) ────────────

AR_KEYWORDS = {
    # Skrip Arab
    "أريد", "أُرِيدُ", "هل", "من", "إلى", "في", "على", "مع",
    "يا", "يَا", "أخي", "أَخِي", "غدا", "غَدًا",
    "الأسبوع", "القادم", "أفضل", "رحلة", "ممكن",
    "احيانا", "أحيانًا", "رمضان",
    # Transliterasi
    "uridu", "bismillah", "alhamdulillah", "insyaallah",
    "inshallah", "masyaallah", "subhanallah", "ramadan",
    "maghrib", "umrah", "hajj", "haram", "madinah",
    "makkah", "jeddah", "saudi",
}

EN_KEYWORDS = {
    "book", "flight", "schedule", "transport", "arrange",
    "explain", "step", "apply", "visa", "travel", "include",
    "visit", "help", "can", "you", "how", "to", "from",
    "the", "and", "for", "with", "after", "simple", "best",
    "direct", "tomorrow", "today", "next", "week", "guide",
    "checklist", "prepare", "translate", "review", "upload",
    "meeting", "submit", "assignment", "deadline", "trip",
    "session", "overwhelmed", "tips", "document", "hotel",
    "budget", "beginner", "online", "email", "attachment",
    "check", "google", "drive", "presentation",
}

ID_KEYWORDS = {
    "saya", "aku", "mau", "ke", "minggu", "depan", "bisa",
    "bantu", "butuh", "tapi", "dari", "dan", "yang", "ini",
    "itu", "ada", "untuk", "dengan", "cara", "pergi", "besok",
    "hari", "pagi", "tolong", "buat", "termasuk", "dibawa",
    "mulai", "belajar", "bahasa", "susah", "gak", "terbatas",
    "dekat", "pilih", "jelaskan", "bagaimana", "kenapa",
    "perbedaan", "secara", "detail", "proses", "sekarang",
    "sudah", "naik", "lanjut", "kapan", "terbaik", "apa",
    "dalam", "bagi", "wajib", "menurut", "kamu", "arab",
    "pemula", "persiapan", "juga", "atau", "udah",
}


# ── Helper ────────────────────────────────────────────────────

def _is_arabic_script(text: str) -> bool:
    for ch in text:
        if '\u0600' <= ch <= '\u06FF':
            return True
    return False


def _apply_map(text: str, mapping: dict) -> str:
    """Terapkan kamus mapping ke teks word by word."""
    words  = text.split()
    result = []
    for w in words:
        punct = w[-1] if w and w[-1] in ".,!?;:" else ""
        clean = w.strip(".,!?;:'\"").lower()
        if clean in mapping:
            result.append(mapping[clean] + punct)
        else:
            result.append(w)
    return " ".join(result)


# ── Normalisasi ───────────────────────────────────────────────

def normalize(raw_text: str) -> str:
    """
    Pipeline normalisasi lengkap:
    1. Unicode NFC — penting untuk karakter Arab
    2. Strip & bersihkan whitespace
    3. _NORM_MAP — koreksi ejaan & penulisan
    4. _CS_TO_ID_MAP — terjemahan CS → Bahasa Indonesia
    5. Hapus karakter kontrol
    """
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", raw_text)

    # 2. Strip & whitespace
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)

    # 3. Koreksi ejaan
    text = _apply_map(text, _NORM_MAP)

    # 4. CS → ID
    text = _apply_map(text, _CS_TO_ID_MAP)

    # 5. Hapus karakter kontrol
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text


# ── Language tagging ──────────────────────────────────────────

def _tag_token(token: str) -> str:
    clean = token.lower().strip(".,!?;:'\"")
    if _is_arabic_script(token):
        return "AR"
    if clean in AR_KEYWORDS:
        return "AR"
    if clean in EN_KEYWORDS and clean not in ID_KEYWORDS:
        return "EN"
    if clean in ID_KEYWORDS and clean not in EN_KEYWORDS:
        return "ID"
    if clean in EN_KEYWORDS and clean in ID_KEYWORDS:
        return "ID"
    return "UNKNOWN"


def tag_languages(text: str) -> list[TextSegment]:
    """
    Tokenisasi → tag per token → resolve UNKNOWN → gabungkan segmen.
    """
    tokens = text.split()
    if not tokens:
        return []

    tagged = [(tok, _tag_token(tok)) for tok in tokens]

    # Resolve UNKNOWN: ikuti bahasa tetangga terdekat
    for i, (tok, lang) in enumerate(tagged):
        if lang == "UNKNOWN":
            left  = next((tagged[j][1] for j in range(i-1, -1, -1)
                          if tagged[j][1] != "UNKNOWN"), None)
            right = next((tagged[j][1] for j in range(i+1, len(tagged))
                          if tagged[j][1] != "UNKNOWN"), None)
            tagged[i] = (tok, left or right or "ID")

    # Gabungkan token berurutan dengan bahasa sama
    segments: list[TextSegment] = []
    cur_tokens = [tagged[0][0]]
    cur_lang   = tagged[0][1]

    for tok, lang in tagged[1:]:
        if lang == cur_lang:
            cur_tokens.append(tok)
        else:
            segments.append(TextSegment(
                text=" ".join(cur_tokens),
                language=cur_lang
            ))
            cur_tokens = [tok]
            cur_lang   = lang

    segments.append(TextSegment(
        text=" ".join(cur_tokens),
        language=cur_lang
    ))
    return segments


def detect_dominant_language(
        segments: list[TextSegment]) -> tuple[str, list[str]]:
    counts: dict[str, int] = {}
    for seg in segments:
        counts[seg.language] = counts.get(seg.language, 0) + len(seg.text.split())
    if not counts:
        return "ID", ["ID"]
    dominant = max(counts, key=counts.get)
    detected = list(counts.keys())
    return dominant, detected


# ── Format untuk LLM ──────────────────────────────────────────

def format_for_llm(normalized: str,
                   segments: list[TextSegment],
                   dominant: str) -> str:
    lang_tags = " + ".join(
        dict.fromkeys(seg.language for seg in segments)
    )
    return (
        f"[CONTEXT: code-switching {lang_tags}, dominant={dominant}]\n"
        f"{normalized}"
    )


# ── Simpan manifest ───────────────────────────────────────────

def save_to_manifest(audio_id: str, result: "ProcessedText") -> None:
    out = {
        "audio_id":           audio_id,
        "raw":                result.raw,
        "normalized":         result.normalized,
        "dominant_language":  result.dominant_language,
        "languages_detected": result.languages_detected,
        "segments": [
            {"text": s.text, "language": s.language}
            for s in result.segments
        ],
        "for_llm": result.for_llm,
    }
    path = MANIFEST_DIR / f"{audio_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Manifest disimpan: {path.name}")


# ── Pipeline utama ────────────────────────────────────────────

def process(stt_text: str,
            audio_id: str = None,
            save_manifest: bool = False) -> ProcessedText:
    """
    Full text preprocessing pipeline:
      raw STT output
        → normalisasi (ejaan + CS→ID)
        → language tagging
        → deteksi bahasa dominan
        → format untuk LLM

    Args:
        stt_text      : output mentah dari whisper.cpp
        audio_id      : ID file audio untuk manifest
        save_manifest : simpan ke data/manifests/
    """
    logger.info(f"Input: \"{stt_text[:80]}\"")

    normalized         = normalize(stt_text)
    segments           = tag_languages(normalized)
    dominant, detected = detect_dominant_language(segments)
    for_llm            = format_for_llm(normalized, segments, dominant)

    result = ProcessedText(
        raw=stt_text,
        normalized=normalized,
        segments=segments,
        dominant_language=dominant,
        languages_detected=detected,
        for_llm=for_llm,
    )

    if save_manifest and audio_id:
        save_to_manifest(audio_id, result)

    logger.info(
        f"dominant={dominant} detected={detected} "
        f"segments={len(segments)}"
    )
    return result