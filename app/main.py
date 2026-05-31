# app/main.py

import os
import time
import shutil
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

from app.utils import get_logger, preprocess_audio, cleanup_temp, AUDIO_DIR
from app.stt import transcribe
from app.text_processor import process
from app.llm import generate_response
from app.tts import synthesize

load_dotenv()

logger = get_logger("main")

app = FastAPI(
    title="Voice Code-Switching System",
    description="Multilingual S2S pipeline: ID + EN + AR",
    version="1.0.0",
)


# ── Health check ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "system": "voice-cs-system"}


@app.get("/health")
def health():
    return {
        "status":  "ok",
        "whisper": "whisper.cpp",
        "llm":     os.getenv("GEMINI_MODEL", "-"),
        "tts":     os.getenv("TTS_MODEL", "-"),
    }


# ── Endpoint utama: audio → audio ─────────────────────────────
@app.post("/pipeline")
async def pipeline(
    file: UploadFile = File(...),
    mode: str = Query(default="normalize", enum=["normalize", "preserve"])):
    """
    Full S2S pipeline:
    Audio input → STT → Text Processing → LLM → TTS → Audio output

    Accepts : WAV / MP3 / M4A
    Returns : WAV audio response
    """
    t_total = time.time()

    # ── Validasi format file ──────────────────────────────────
    allowed = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    suffix  = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: {suffix}. Gunakan: {allowed}"
        )

    # ── Simpan file upload ke AUDIO_DIR sementara ─────────────
    upload_path = AUDIO_DIR / f"upload_{int(time.time())}{suffix}"
    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Upload diterima: {upload_path.name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal simpan upload: {e}")

    timings = {}

    try:
        # ── Step 1: STT ───────────────────────────────────────
        t0         = time.time()
        stt_result = transcribe(upload_path, cleanup=True)
        timings["stt_s"] = round(time.time() - t0, 2)

        if not stt_result["text"].strip():
            raise HTTPException(
                status_code=422,
                detail="STT tidak menghasilkan teks. Periksa kualitas audio."
            )

        logger.info(f"STT: \"{stt_result['text'][:60]}\"")

        # ── Step 2: Text processing ───────────────────────────
        t0         = time.time()
        tp_result  = process(stt_result["text"])
        timings["text_proc_s"] = round(time.time() - t0, 2)

        logger.info(
            f"TextProc: dominant={tp_result.dominant_language} "
            f"langs={tp_result.languages_detected}"
        )

        # ── Step 3: LLM ───────────────────────────────────────
        t0         = time.time()
        llm_result = generate_response(
            tp_result.for_llm,
            tp_result.dominant_language,
            mode=mode
        )
        timings["llm_s"] = round(time.time() - t0, 2)

        if not llm_result["success"]:
            raise HTTPException(
                status_code=503,
                detail=f"LLM gagal: {llm_result['response_text']}"
            )

        logger.info(f"LLM: \"{llm_result['response_text'][:60]}\"")

        # ── Step 4: TTS ───────────────────────────────────────
        t0         = time.time()
        audio_id   = stt_result.get("audio_id", f"resp_{int(time.time())}")
        tts_result = synthesize(
            response_text=llm_result["response_text"],
            dominant_language=tp_result.dominant_language,
            audio_id=audio_id,
        )
        timings["tts_s"] = round(time.time() - t0, 2)

        if not tts_result["success"] or not tts_result["output_path"]:
            raise HTTPException(
                status_code=503,
                detail="TTS gagal menghasilkan audio."
            )

        timings["total_s"] = round(time.time() - t_total, 2)

        logger.info(
            f"Pipeline selesai | "
            f"stt={timings['stt_s']}s "
            f"llm={timings['llm_s']}s "
            f"tts={timings['tts_s']}s "
            f"total={timings['total_s']}s"
        )

        # ── Return audio response ─────────────────────────────
        
        # Encode teks agar huruf Arab dan Enter (\n) tidak bikin server crash
        safe_stt = urllib.parse.quote(stt_result["text"])
        safe_llm = urllib.parse.quote(llm_result["response_text"])
        safe_lang = urllib.parse.quote(tp_result.dominant_language)

        return FileResponse(
            path=str(tts_result["output_path"]),
            media_type="audio/wav",
            filename=f"response_{audio_id}.wav",
            headers={
                "X-STT-Text":       safe_stt,
                "X-LLM-Response":   safe_llm,
                "X-Dominant-Lang":  safe_lang,
                "X-Latency-STT":    str(timings["stt_s"]),
                "X-Latency-LLM":    str(timings["llm_s"]),
                "X-Latency-TTS":    str(timings["tts_s"]),
                "X-Latency-Total":  str(timings["total_s"]),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Hapus file upload setelah selesai
        cleanup_temp(upload_path)

@app.post("/voice-chat")
async def voice_chat(
    file: UploadFile = File(...),
    mode: str = Query(default="normalize", enum=["normalize", "preserve"])
):
    """
    Endpoint utama sesuai spesifikasi aslab.
    mode=normalize : respons satu bahasa dominan (default)
    mode=preserve  : respons pertahankan code-switching
    """
    return await pipeline(file, mode)


# ── Endpoint: teks saja (tanpa audio output) ──────────────────
@app.post("/pipeline/text")
async def pipeline_text(file: UploadFile = File(...)):
    """
    Pipeline tanpa TTS — return JSON teks saja.
    Berguna untuk debug dan evaluasi.
    """
    t_total = time.time()
    suffix  = Path(file.filename).suffix.lower()
    upload_path = AUDIO_DIR / f"upload_{int(time.time())}{suffix}"

    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        stt_result = transcribe(upload_path, cleanup=True)
        tp_result  = process(stt_result["text"])
        llm_result = generate_response(
            tp_result.for_llm,
            tp_result.dominant_language
        )

        return JSONResponse({
            "stt_text":          stt_result["text"],
            "normalized":        tp_result.normalized,
            "dominant_language": tp_result.dominant_language,
            "languages_detected":tp_result.languages_detected,
            "segments": [
                {"text": s.text, "language": s.language}
                for s in tp_result.segments
            ],
            "llm_response":      llm_result["response_text"],
            "llm_success":       llm_result["success"],
            "latency_total_s":   round(time.time() - t_total, 2),
        })

    except Exception as e:
        logger.error(f"pipeline/text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cleanup_temp(upload_path)


# ── Jalankan server ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )