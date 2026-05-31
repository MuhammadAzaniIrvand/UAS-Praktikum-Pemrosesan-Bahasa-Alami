import os
import requests
import urllib.parse
import tempfile
import gradio as gr

API_URL = "http://127.0.0.1:8000/voice-chat"

def run_client(mic_audio, upload_audio, mode):
    audio_path = mic_audio or upload_audio
    
    if not audio_path:
        return None, "Tidak ada input audio.", "", "", "Harap masukkan audio."

    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            
            # mode dikirim via parameter URL (Query parameter)
            params = {"mode": mode}
            
            # Request post ke FastAPI
            res = requests.post(API_URL, files=files, params=params)
            res.raise_for_status()
            
            # 1. Ambil teks dari Headers dan kembalikan (decode) ke teks asli
            stt_text = urllib.parse.unquote(res.headers.get("X-STT-Text", ""))
            llm_text = urllib.parse.unquote(res.headers.get("X-LLM-Response", ""))
            dom_lang = urllib.parse.unquote(res.headers.get("X-Dominant-Lang", ""))
            
            # Ambil data latensi untuk kotak Info
            lat_stt = res.headers.get("X-Latency-STT", "0")
            lat_llm = res.headers.get("X-Latency-LLM", "0")
            lat_tts = res.headers.get("X-Latency-TTS", "0")
            lat_tot = res.headers.get("X-Latency-Total", "0")
            
            info_text = (
                f"Bahasa Dominan: {dom_lang}\n"
                f"Latensi STT: {lat_stt}s | LLM: {lat_llm}s | TTS: {lat_tts}s\n"
                f"Total Waktu: {lat_tot}s"
            )

            # 2. Simpan respon audio (binary) ke file WAV lokal sementara agar Gradio bisa putar
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio.write(res.content)
            temp_audio.close()
            
            return (
                temp_audio.name, 
                stt_text, 
                f"Bahasa dominan: {dom_lang}", 
                llm_text, 
                info_text
            )

    except Exception as e:
        err = f"Error: {str(e)}"
        return None, err, "", "", err


with gr.Blocks(title="Voice Code-Switching System") as demo:
    gr.Markdown("# 🎙️ Voice Code-Switching System\n**Multilingual Speech-to-Speech**")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.Tab("🎤 Rekam"):
                    audio_mic = gr.Audio(type="filepath", sources=["microphone"])
                with gr.Tab("📁 Upload"):
                    audio_upload = gr.Audio(type="filepath", sources=["upload"])
            
            mode_radio = gr.Radio(
                choices=["normalize", "preserve"],
                value="normalize",
                label="Mode Output"
            )
            run_btn = gr.Button("▶ Jalankan Pipeline", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="🔊 Output Audio", type="filepath")
            info_output = gr.Textbox(label="ℹ️ Info", lines=5, interactive=False)

    with gr.Row():
        with gr.Column():
            stt_output = gr.Textbox(label="📝 STT", lines=2, interactive=False)
            normalized_output = gr.Textbox(label="🔄 Text Processing", lines=2, interactive=False)
        with gr.Column():
            llm_output = gr.Textbox(label="🤖 LLM Respons", lines=4, interactive=False)

    run_btn.click(
        fn=run_client,
        inputs=[audio_mic, audio_upload, mode_radio],
        outputs=[audio_output, stt_output, normalized_output, llm_output, info_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)