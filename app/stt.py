import os
from faster_whisper import WhisperModel

# Model Whisper dimuat sekali saat aplikasi dijalankan
model = WhisperModel(
    "large-v3",
    device="cpu",   # ganti "cuda" kalau pakai GPU NVIDIA
    compute_type="int8"
)

def transcribe(audio_path: str):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {audio_path}"
        )

    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    result=[]

    for segment in segments:
        result.append(segment.text)

    full_text=" ".join(result)

    return {
        "language": info.language,
        "text": full_text.strip()
    }