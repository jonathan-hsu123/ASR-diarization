import argparse
import shutil
import wget
from pathlib import Path

dataset_url = "https://openslr.trmal.net/resources/12/test-clean.tar.gz"

def preprocess_dataset(output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    filename = wget.download(dataset_url, out=str(output_path))
    shutil.unpack_archive(filename, extract_dir=output_path)

    dataset_path = output_path / Path("LibriSpeech/test-clean")
    nested = sorted(p for p in dataset_path.rglob("*")
                if p.is_file() and p.parent != dataset_path)
    txt_files = [p for p in nested if p.suffix.lower() == ".txt"]
    other_files = [p for p in nested if p.suffix.lower() != ".txt"]

    ASR_dir_path = output_path / Path("processed_ASR")
    transcript_path = ASR_dir_path / "ASR_transcript.txt"
    ASR_dir_path.mkdir(parents=True, exist_ok=True)

    with open(transcript_path, "w") as f:
        for txt_file in txt_files:
            with open(txt_file, "r") as f2:
                f.write(f2.read())

    for other_file in other_files:
        shutil.move(str(other_file), str(ASR_dir_path / other_file.name))

    shutil.rmtree(output_path / "LibriSpeech")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='data')
    args = parser.parse_args()
    preprocess_dataset(Path(args.output_path))