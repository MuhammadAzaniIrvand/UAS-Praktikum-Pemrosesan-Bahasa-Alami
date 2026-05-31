import re
import json
import time
from pathlib import Path

from app.utils import get_logger, AUDIO_DIR, BASE_DIR
from app.stt import transcribe
from app.text_processor import process
from app.llm import generate_response
from app.tts import synthesize

logger      = get_logger("analisis_pipeline")
RESULTS_DIR = BASE_DIR / "results" / "pipeline_output"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TTS_OUTPUT_DIR = BASE_DIR / "results" / "tts_output"
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = RESULTS_DIR / "pipeline_summary.json"


# ── Natural sort (audio1, audio2, ..., audio10, audio11) ──────
def natural_key(path: Path) -> list:
    return [
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r'(\d+)', path.stem)
    ]


def load_existing_results() -> dict:
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: dict) -> None:
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def is_done(audio_id: str, results: dict) -> bool:
    r = results.get(audio_id, {})
    return r.get("tts_success", False)


def run_pipeline(audio_path: Path, audio_id: str) -> dict:
    result = {
        "audio_id":   audio_id,
        "audio_path": str(audio_path),
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── STT ───────────────────────────────────────────────────
    try:
        stt = transcribe(audio_path, cleanup=True)
        result.update({
            "stt_text":    stt["text"],
            "stt_latency": stt["latency_s"],
            "stt_success": True,
        })
        logger.info(f"[{audio_id}] STT: \"{stt['text'][:60]}\"")
    except Exception as e:
        result.update({"stt_success": False, "stt_error": str(e)})
        logger.error(f"[{audio_id}] STT gagal: {e}")
        return result

    # ── Text processing ───────────────────────────────────────
    tp = process(stt["text"], audio_id=audio_id, save_manifest=True)
    result.update({
        "normalized":         tp.normalized,
        "dominant_language":  tp.dominant_language,
        "languages_detected": tp.languages_detected,
        "segments": [
            {"text": s.text, "language": s.language}
            for s in tp.segments
        ],
    })

    # ── LLM ───────────────────────────────────────────────────
    try:
        llm = generate_response(tp.for_llm, tp.dominant_language)
        result.update({
            "llm_response": llm["response_text"],
            "llm_latency":  llm["latency_s"],
            "llm_success":  llm["success"],
        })
        logger.info(f"[{audio_id}] LLM: \"{llm['response_text'][:60]}\"")
    except Exception as e:
        result.update({"llm_success": False, "llm_error": str(e)})
        logger.error(f"[{audio_id}] LLM gagal: {e}")
        return result

    # ── TTS — simpan ke folder terpisah ───────────────────────
    try:
        tts_out = TTS_OUTPUT_DIR / f"response_{audio_id}.wav"
        tts = synthesize(
            response_text=llm["response_text"],
            dominant_language=tp.dominant_language,
            audio_id=audio_id,
        )
        # Pindahkan ke TTS_OUTPUT_DIR kalau synthesize simpan di tempat lain
        if tts["output_path"] and Path(tts["output_path"]) != tts_out:
            Path(tts["output_path"]).rename(tts_out)

        result.update({
            "tts_output":  str(tts_out),
            "tts_latency": tts["latency_s"],
            "tts_success": tts["success"],
        })
        logger.info(f"[{audio_id}] TTS: {tts_out}")
    except Exception as e:
        result.update({"tts_success": False, "tts_error": str(e)})
        logger.error(f"[{audio_id}] TTS gagal: {e}")

    result["total_latency"] = round(
        result.get("stt_latency", 0) +
        result.get("llm_latency", 0) +
        result.get("tts_latency", 0), 2
    )
    return result


def main():
    print("=" * 65)
    print("ANALISIS PIPELINE — SELURUH CORPUS AUDIO")
    print("=" * 65)

    all_results = load_existing_results()
    skipped     = sum(1 for r in all_results.values() if r.get("tts_success"))
    if skipped:
        print(f"Resume: {skipped} audio sudah selesai, skip.")

    # Natural sort — urut berdasarkan angka
    audio_files = sorted(
        [f for f in AUDIO_DIR.glob("*.wav")
         if not f.stem.startswith("prep_")
         and not f.stem.startswith("response_")],
        key=natural_key
    )

    print(f"Total audio ditemukan : {len(audio_files)}")
    print(f"Output JSON           : {SUMMARY_FILE}")
    print(f"Output TTS            : {TTS_OUTPUT_DIR}")
    print("-" * 65)

    for i, audio_path in enumerate(audio_files, 1):
        audio_id = audio_path.stem

        if is_done(audio_id, all_results):
            print(f"[{i}/{len(audio_files)}] [{audio_id}] SKIP")
            continue

        print(f"\n[{i}/{len(audio_files)}] [{audio_id}] Memproses...")
        result = run_pipeline(audio_path, audio_id)

        all_results[audio_id] = result
        save_results(all_results)

        stt_ok  = result.get("stt_success", False)
        llm_ok  = result.get("llm_success", False)
        tts_ok  = result.get("tts_success", False)
        latency = result.get("total_latency", 0)

        print(f"  STT : {'✓' if stt_ok else '✗'} | {result.get('stt_text','')[:50]}")
        print(f"  LANG: {result.get('dominant_language','-')} {result.get('languages_detected','-')}")
        print(f"  LLM : {'✓' if llm_ok else '✗'} | {result.get('llm_response','')[:50]}")
        print(f"  TTS : {'✓' if tts_ok else '✗'} | {result.get('tts_output','')}")
        print(f"  TIME: {latency}s total")

        if llm_ok:
            time.sleep(3)

    # ── Summary ───────────────────────────────────────────────
    done    = [r for r in all_results.values() if r.get("tts_success")]
    stt_ok  = [r for r in all_results.values() if r.get("stt_success")]
    llm_ok  = [r for r in all_results.values() if r.get("llm_success")]

    avg_stt = round(sum(r.get("stt_latency", 0) for r in stt_ok) / len(stt_ok), 2) if stt_ok else 0
    avg_llm = round(sum(r.get("llm_latency", 0) for r in llm_ok) / len(llm_ok), 2) if llm_ok else 0
    avg_tot = round(sum(r.get("total_latency", 0) for r in done) / len(done), 2)    if done    else 0

    print("\n" + "=" * 65)
    print("SUMMARY")
    print(f"  Total audio      : {len(all_results)}")
    print(f"  Pipeline selesai : {len(done)}/{len(all_results)}")
    print(f"  STT berhasil     : {len(stt_ok)}/{len(all_results)}")
    print(f"  LLM berhasil     : {len(llm_ok)}/{len(all_results)}")
    print(f"  Avg STT latency  : {avg_stt}s")
    print(f"  Avg LLM latency  : {avg_llm}s")
    print(f"  Avg total        : {avg_tot}s")
    print(f"\n  JSON  : {SUMMARY_FILE}")
    print(f"  Audio : {TTS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
