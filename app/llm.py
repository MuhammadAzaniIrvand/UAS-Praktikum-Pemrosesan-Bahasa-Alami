# app/llm.py

import os
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.utils import get_logger

load_dotenv()

logger       = get_logger("llm")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemma-4-26b-a4b-it")

# ── Multiple API keys ─────────────────────────────────────────
_API_KEYS = [
    k for k in [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3"),
    ] if k
]

if not _API_KEYS:
    logger.warning("Tidak ada GEMINI_API_KEY di .env")

_key_index = 0
_clients   = {}


def get_client() -> genai.Client:
    global _key_index
    key = _API_KEYS[_key_index % len(_API_KEYS)]
    if key not in _clients:
        _clients[key] = genai.Client(api_key=key)
        logger.info(
            f"Client initialized | key_index={_key_index} | model={GEMINI_MODEL}"
        )
    return _clients[key]


def rotate_key() -> bool:
    global _key_index
    _key_index += 1
    if _key_index >= len(_API_KEYS):
        logger.error("Semua API key exhausted.")
        return False
    logger.info(f"Rotating ke key_index={_key_index}")
    return True


_SYSTEM_INSTRUCTIONS = {
    "normalize": """Kamu adalah asisten perjalanan umrah dan haji yang ramah.
Aturan wajib:
1. Deteksi bahasa dominan dari input dan gunakan bahasa yang sama dalam respons.
2. Respons menggunakan SATU bahasa dominan saja — jangan campur bahasa.
3. Respons singkat dan praktis — maksimal 3 kalimat.
4. JANGAN ulangi pertanyaan pengguna.
5. JANGAN gunakan bullet point atau numbering.
6. JANGAN sertakan label seperti "Jawaban:" atau sejenisnya.
7. Untuk topik umrah/haji/perjalanan Saudi, berikan informasi yang akurat.
""",
    "preserve": """Kamu adalah asisten perjalanan umrah dan haji yang ramah.
Aturan wajib:
1. Deteksi bahasa dominan dari input dan gunakan bahasa yang sama dalam respons.
2. Pertahankan pola code-switching secara natural — sisipkan kata dari bahasa lain seperti penutur asli.
3. Contoh: "Oke, untuk pesan penerbangan ke Jeddah, kamu bisa check di Traveloka atau website maskapai."
4. Respons singkat dan praktis — maksimal 3 kalimat.
5. JANGAN ulangi pertanyaan pengguna.
6. JANGAN gunakan bullet point atau numbering.
7. JANGAN sertakan label seperti "Jawaban:" atau sejenisnya.
""",
}

# ── Build user message ────────────────────────────────────────
def _build_user_message(normalized_text: str,
                        dominant: str) -> str:
    """
    Bangun user message dari teks yang sudah dinormalisasi
    oleh text_processor (CS→ID sudah diterapkan).
    """
    lang_hint = {
        "ID": "Bahasa dominan: Indonesia. Jawab dalam Bahasa Indonesia.",
        "EN": "Dominant language: English. Please answer in English.",
        "AR": "اللغة المهيمنة: العربية. أجب باللغة العربية.",
    }.get(dominant, "Bahasa dominan: Indonesia. Jawab dalam Bahasa Indonesia.")

    return f"{lang_hint}\n\nPertanyaan: {normalized_text}"


# ── Generate response ─────────────────────────────────────────
def generate_response(processed_text_for_llm: str,
                      dominant_language: str = "ID",
                      mode : str = "normalize",
                      max_retries: int = 2) -> dict:
    """
    Kirim teks yang sudah dinormalisasi ke Gemma.

    Args:
        processed_text_for_llm : output dari text_processor.format_for_llm()
                                 (sudah melalui CS→ID normalization)
        dominant_language      : bahasa dominan (ID/EN/AR)
        max_retries            : jumlah retry per key

    Returns:
        {
          "response_text" : str,
          "latency_s"     : float,
          "model"         : str,
          "success"       : bool
        }
    """
    global _key_index
    t0 = time.time()

    # Pisahkan context tag dari teks normalized
    lines     = processed_text_for_llm.strip().split("\n")
    norm_text = processed_text_for_llm

    if lines and lines[0].startswith("[CONTEXT:"):
        norm_text = "\n".join(lines[1:]).strip()

    user_message = _build_user_message(norm_text, dominant_language)

    keys_tried = 0
    while keys_tried < len(_API_KEYS):
        client  = get_client()
        attempt = 0

        while attempt < max_retries + 1:
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM_INSTRUCTIONS[mode],
                        temperature=0.7,
                        max_output_tokens=1024,
                    ),
                )

                latency       = round(time.time() - t0, 2)
                response_text = None

                # Cara 1: response.text langsung
                try:
                    if response.text:
                        response_text = response.text.strip()
                except Exception:
                    pass

                # Cara 2: dari candidates → parts
                if not response_text and response.candidates:
                    candidate = response.candidates[0]
                    logger.info(f"finish_reason={candidate.finish_reason}")
                    if hasattr(candidate, "content") and candidate.content:
                        parts_text = ""
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                parts_text += part.text
                        if parts_text.strip():
                            response_text = parts_text.strip()

                if not response_text:
                    raise ValueError("Response text kosong")

                response_text = _clean_response(response_text)

                if not response_text:
                    logger.warning("Echo terdeteksi, retry...")
                    raise ValueError("Echo response")

                logger.info(
                    f"OK | key={_key_index} | latency={latency}s | "
                    f"response=\"{response_text[:60]}\""
                )

                return {
                    "response_text": response_text,
                    "latency_s":     latency,
                    "model":         GEMINI_MODEL,
                    "success":       True,
                }

            except Exception as e:
                err = str(e)
                attempt += 1

                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    match = re.search(r'retryDelay.*?(\d+)s', err)
                    wait  = int(match.group(1)) if match else 30
                    logger.warning(
                        f"Rate limit key={_key_index} | wait={wait}s"
                    )
                    if len(_API_KEYS) > 1:
                        if rotate_key():
                            break
                    logger.info(f"Tidak ada key lain, tunggu {wait}s...")
                    time.sleep(wait)

                elif "quota" in err.lower():
                    logger.warning(
                        f"Quota habis key={_key_index}, rotate..."
                    )
                    if not rotate_key():
                        keys_tried = len(_API_KEYS)
                        break

                else:
                    logger.warning(f"Attempt {attempt}: {err[:100]}")
                    if attempt <= max_retries:
                        time.sleep(1.5 * attempt)

        keys_tried += 1

    latency = round(time.time() - t0, 2)
    logger.error("Semua API key exhausted.")
    return {
        "response_text": _fallback_response(dominant_language),
        "latency_s":     latency,
        "model":         GEMINI_MODEL,
        "success":       False,
    }


def _clean_response(text: str) -> str:
    # Hapus prefix label
    text = re.sub(
        r'^(Jawaban|Answer|Respons|Response)\s*:\s*',
        '', text, flags=re.IGNORECASE
    )
    # Hapus bullet point
    text = re.sub(r'^\s*[\*\-]\s+', '', text, flags=re.MULTILINE)
    # Bersihkan baris kosong
    text = re.sub(r'\n{2,}', '\n', text).strip()

    # Deteksi echo
    echo_markers = [
        "Aturan wajib", "system_instruction",
        "Dominant language:", "Bahasa dominan:",
        "[CONTEXT", "dominant=",
    ]
    for marker in echo_markers:
        if marker.lower() in text.lower():
            match = re.search(
                r'(Jawaban|Answer)\s*:\s*(.+)',
                text, flags=re.IGNORECASE | re.DOTALL
            )
            text = match.group(2).strip() if match else ""
            break

    # Potong di kalimat lengkap terakhir
    if len(text) > 400:
        for punct in ['.', '?', '!']:
            cutoff = text[:400].rfind(punct)
            if cutoff > 0:
                text = text[:cutoff + 1]
                break

    return text


def _fallback_response(dominant_language: str) -> str:
    return {
        "ID": "Maaf, saya sedang tidak bisa memproses permintaan. Silakan coba lagi.",
        "EN": "Sorry, I'm unable to process your request right now. Please try again.",
        "AR": "عذراً، لا أستطيع معالجة طلبك الآن. يرجى المحاولة مرة أخرى.",
    }.get(dominant_language, "Maaf, saya sedang tidak bisa memproses permintaan.")


def process_stt_result(stt_result: dict) -> dict:
    for_llm  = stt_result.get("for_llm", stt_result.get("text", ""))
    dominant = stt_result.get("dominant_language", "ID")
    return generate_response(for_llm, dominant)