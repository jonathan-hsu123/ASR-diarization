import yaml
import argparse
import shutil
import wget
from pathlib import Path

config_yaml_path = 'dataset.yaml'
supported_task = ['ASR', 'Diarization']

def download_dataset(task, output_path):
    assert task in supported_task, f"The task is not supported. Current support task: {supported_task}."
    with open(config_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        dataset_dict = config[task]
        Dataset_url = dataset_dict['Dataset_url']
        filename = wget.download(Dataset_url, out=output_path)
        shutil.unpack_archive(filename, extract_dir=output_path)

def preprocess_dataset(task, output_path):
    download_dataset(task, output_path)

    if task == 'ASR':
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

        shutil.rmtree("LibriSpeech")

    elif task == 'Diarization':
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='ASR')
    parser.add_argument('--output_path', default='./')
    args = parser.parse_args()
    preprocess_dataset(args.task, args.output_path)