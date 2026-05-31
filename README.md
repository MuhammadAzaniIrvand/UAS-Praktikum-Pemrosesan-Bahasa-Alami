# 🎙️ Voice Code-Switching System

Sistem multilingual **Speech-to-Speech (S2S)** end-to-end yang dirancang untuk memproses ujaran **code-switching** yang melibatkan **Bahasa Indonesia (ID)**, **Bahasa Inggris (EN)**, dan **Bahasa Arab (AR)**.

---

## 📌 Fitur Utama

- 🎤 Speech-to-Text menggunakan **whisper.cpp**
- 📝 Normalisasi teks dan language tagging
- 🤖 Pemrosesan bahasa menggunakan **Gemma 4** melalui Google Gemini API
- 🔊 Text-to-Speech menggunakan **Coqui TTS** dan **Wikidepia VITS**
- 🌐 Antarmuka pengguna berbasis **Gradio**
- ⚡ Backend API menggunakan **FastAPI**

---

## 🔄 Arsitektur Pipeline

```text
Audio Input
      │
      ▼
STT (whisper.cpp)
      │
      ▼
Text Processing
      │
      ▼
LLM (Gemma API)
      │
      ▼
TTS (Coqui / Wikidepia)
      │
      ▼
Audio Output
```

---

## 📁 Struktur Direktori

```text
voice-cs-system/
├── app/
│   ├── main.py              # FastAPI endpoint (Backend)
│   ├── stt.py               # Modul Speech-to-Text (whisper.cpp)
│   ├── llm.py               # Modul LLM (Gemma via Gemini API)
│   ├── tts.py               # Modul Text-to-Speech (Coqui TTS)
│   ├── text_processor.py    # Modul Normalisasi & Language Tagging
│   ├── utils.py             # Utilitas path, logger, dan audio prep
│   └── coqui_tts/           # Model Wikidepia VITS
│
├── data/
│   ├── corpus/              # Dataset audio dan ground truth
│   └── manifests/           # Hasil text processing
│
├── models/
│   └── whisper.cpp/         # Binary dan model ggml
│
├── gradio_app/
│   └── app.py               # Frontend Gradio
│
├── log/                     # Log sistem
├── results/                 # Hasil evaluasi
│
├── analisis_pipeline.py     # Evaluasi seluruh corpus
├── .env                     # Konfigurasi environment
├── requirements.txt         # Dependensi Python
└── README.md
```

---

## ⚙️ Persiapan dan Instalasi

### 1. Membuat Virtual Environment

Pastikan menggunakan **Python 3.11**.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Konfigurasi Environment

Buat file `.env` pada root project.

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_API_KEY_2=your_second_key_here

GEMINI_MODEL=models/gemma-4-26b-a4b-it

WHISPER_MODEL_FILE=ggml-medium.bin

TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
```

---

### 3. Setup STT (whisper.cpp)

Build `whisper.cpp` dan unduh model **medium**.

```bash
cd models/whisper.cpp

cmake -B build
cmake --build build --config Release

bash models/download-ggml-model.sh medium
```

---

### 4. Setup Model TTS (Wikidepia VITS)

Unduh checkpoint dan konfigurasi model.

```bash
wget -P app/coqui_tts/ \
https://github.com/Wikidepia/indonesian-tts/releases/download/v1.2/checkpoint_1260000-inference.pth

wget -P app/coqui_tts/ \
https://github.com/Wikidepia/indonesian-tts/releases/download/v1.2/config.json

wget -P app/coqui_tts/ \
https://github.com/Wikidepia/indonesian-tts/releases/download/v1.2/speakers.pth
```

---

## 🚀 Menjalankan Sistem

Sistem menggunakan arsitektur **Client-Server**, sehingga backend dan frontend dijalankan pada terminal yang berbeda.

### 1. Menjalankan Backend (FastAPI)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend akan berjalan pada:

```text
http://localhost:8000
```

---

### 2. Menjalankan Frontend (Gradio)

```bash
python gradio_app/app.py
```

Frontend akan berjalan pada:

```text
http://localhost:7860
```

---

### 3. Evaluasi Pipeline (Opsional)

Untuk melakukan evaluasi terhadap seluruh corpus:

```bash
python analisis_pipeline.py
```

---

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi |
|-----------|-----------|
| Speech-to-Text (STT) | whisper.cpp (ggml-medium) |
| Text Processing | Custom Python NLP Module |
| Large Language Model | Gemma 4 (Google Gemini API) |
| Text-to-Speech (ID) | Wikidepia VITS |
| Text-to-Speech (EN/AR) | Coqui XTTS v2 |
| Backend | FastAPI + Uvicorn |
| Frontend | Gradio |

---

## 📊 Alur Pemrosesan

1. Pengguna mengunggah atau merekam audio.
2. Audio dikonversi menjadi teks menggunakan **whisper.cpp**.
3. Teks dinormalisasi dan dilakukan **language tagging**.
4. Teks diproses oleh **Gemma 4** untuk menghasilkan respons.
5. Respons diubah kembali menjadi suara menggunakan **Coqui TTS** atau **Wikidepia VITS**.
6. Audio hasil sintesis dikirim kembali ke pengguna.

---

## 👨‍💻 Pengembang

Proyek ini dikembangkan sebagai sistem penelitian dan implementasi **Speech-to-Speech Multilingual Code-Switching** untuk Bahasa Indonesia, Inggris, dan Arab.
