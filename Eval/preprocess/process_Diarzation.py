from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
import argparse

REPO_ID = "ggfox00000/dia-voxconverse-test"
DEFAULT_REVISION = "7e3ea2807400b9ed79451e20513f7a50ac4fd846"

def preprocess_voxconverse(output_dir, revision):
    api = HfApi()
    revision = revision or api.dataset_info(REPO_ID).sha
    print("revision:", revision)

    files = api.list_repo_files(REPO_ID, repo_type="dataset", revision=revision)
    stems = sorted(Path(f).stem for f in files if f.endswith(".wav"))
    patterns = [f"audio/test/{s}.wav" for s in stems] + [f"rttm/test/{s}.rttm" for s in stems]

    snapshot_download(REPO_ID, repo_type="dataset", revision=revision,
                      allow_patterns=patterns, local_dir=output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='../../data/processed_Diar')
    parser.add_argument('--revision', default=DEFAULT_REVISION)
    args = parser.parse_args()
    preprocess_voxconverse(Path(args.output_path), args.revision)