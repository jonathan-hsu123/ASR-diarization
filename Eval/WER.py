from jiwer import wer
import re
from pathlib import Path
import argparse
from faster_whisper import WhisperModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed_ASR"

def normalize(text):
    return re.sub(r"[^\w\s]", "", text).lower()

def calculate_WER(audio_dir_path, transcript_path, model_size, device, compute_type, limit=None):
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    predictions = []
    transcripts = []
    with open(transcript_path, "r") as f:
        lines = f.readlines()[:limit]
    for line in lines:
        audio_ID, transcript = line.split(maxsplit=1)
        audio_path = audio_dir_path / f"{audio_ID}.flac"

        segments, info = model.transcribe(str(audio_path), beam_size=5)
        prediction = " ".join(segment.text for segment in segments)
        predictions.append(normalize(prediction))
        transcripts.append(normalize(transcript))
    return wer(transcripts, predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir_path', type=Path, default=DATA_DIR)
    parser.add_argument('--transcript_path', type=Path, default=DATA_DIR / 'ASR_transcript.txt')
    parser.add_argument('--model_size', default='tiny')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--compute_type', default='int8')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    result = calculate_WER(args.audio_dir_path, args.transcript_path, args.model_size,
                           args.device, args.compute_type, args.limit)
    print(f"WER = {result:.4f}")