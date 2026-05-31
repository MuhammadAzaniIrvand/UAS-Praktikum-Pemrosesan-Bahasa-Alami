from pathlib import Path
from jiwer import wer, cer
import re

from app.stt import transcribe
from app.utils import (
    normalize_text,
    preprocess_audio
)

AUDIO_DIR = Path(
    "data/corpus/audio"
)

TRANSCRIPT_DIR = Path(
    "data/corpus/transcripts"
)

OUTPUT_FILE = Path(
    "results/stt_evaluation.txt"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


def evaluate():

    results=[]

    audio_files=AUDIO_DIR.glob(
        "*.wav"
    )

    for audio in audio_files:

        try:

            audio_name=audio.stem

            audio_id=audio_name.split(
                "_"
            )[1]

            number=re.search(
                r"\d+",
                audio_id
            ).group()

            number=int(
                number
            )

            audio_id=f"audio{number}"

            transcript_file=(
                TRANSCRIPT_DIR/
                f"{audio_id}.txt"
            )

            if not transcript_file.exists():

                print(
                    f"Transcript tidak ditemukan: {transcript_file}"
                )

                continue


            with open(
                transcript_file,
                "r",
                encoding="utf-8"
            ) as f:

                ground_truth=f.read()


            processed_audio=preprocess_audio(
                audio
            )


            prediction=transcribe(
                processed_audio
            )


            ground_truth=normalize_text(
                ground_truth
            )

            prediction=normalize_text(
                prediction
            )


            result={

                "audio":
                audio.name,

                "truth":
                ground_truth,

                "prediction":
                prediction,

                "wer":
                round(
                    wer(
                        ground_truth,
                        prediction
                    ),
                    4
                ),

                "cer":
                round(
                    cer(
                        ground_truth,
                        prediction
                    ),
                    4
                )
            }

            results.append(
                result
            )

        except Exception as e:

            print(
                f"Gagal memproses {audio.name}: {e}"
            )

    return results


if __name__=="__main__":

    output=evaluate()

    avg_wer=sum(
        r["wer"]
        for r in output
    )/len(output)

    avg_cer=sum(
        r["cer"]
        for r in output
    )/len(output)


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "===== HASIL STT EVALUATION =====\n\n"
        )

        for row in output:

            text=(
                f"{'='*60}\n"
                f"Audio : {row['audio']}\n\n"
                f"Truth :\n"
                f"{row['truth']}\n\n"
                f"Prediction :\n"
                f"{row['prediction']}\n\n"
                f"WER : {row['wer']}\n"
                f"CER : {row['cer']}\n\n"
            )

            f.write(
                text
            )


        f.write(
            "="*60+"\n"
        )

        f.write(
            "BASELINE EVALUATION\n"
        )

        f.write(
            f"Jumlah audio : {len(output)}\n"
        )

        f.write(
            f"Rata-rata WER : {avg_wer:.4f}\n"
        )

        f.write(
            f"Rata-rata CER : {avg_cer:.4f}\n"
        )


    print(
        f"\nHasil disimpan ke: {OUTPUT_FILE}"
    )