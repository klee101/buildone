"""Paths shared by setup and the offline inference workers."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAM_FILE = "sam3.1_multiplex.pt"
QWEN_NAME = "Qwen-Image-2.1-Turbo"


def weight_path(variable, default):
    path = Path(os.getenv(variable) or default).expanduser()
    return (ROOT / path).resolve()


def sam_weights():
    return weight_path("BUILDONE_SAM_WEIGHTS", Path("models/sam3.1") / SAM_FILE)


def qwen_weights():
    return weight_path("BUILDONE_QWEN_WEIGHTS", Path("models") / QWEN_NAME)


def prepare(include_qwen):
    from huggingface_hub import get_token, hf_hub_download, login, snapshot_download

    checkpoint = sam_weights()
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        if os.getenv("BUILDONE_SAM_WEIGHTS"):
            raise SystemExit(f"BUILDONE_SAM_WEIGHTS must point to an existing checkpoint: {checkpoint}")
        import sys
        if not get_token():
            print("Request access at https://huggingface.co/facebook/sam3.1 and use a read token "
                  "from https://huggingface.co/settings/tokens after approval.", flush=True)
            if not sys.stdin.isatty():
                raise SystemExit("Set HF_TOKEN or run .venv/bin/hf auth login, then rerun setup.")
            login()
        hf_hub_download("facebook/sam3.1", SAM_FILE, local_dir=checkpoint.parent)
    print(f"SAM checkpoint: {checkpoint}", flush=True)
    if include_qwen:
        directory = qwen_weights()
        if os.getenv("BUILDONE_QWEN_WEIGHTS"):
            if not (directory / "model_index.json").is_file():
                raise SystemExit(f"Expected a local Diffusers snapshot in {directory}")
        else:
            snapshot_download(f"Qwen/{QWEN_NAME}", local_dir=directory, ignore_patterns=["assets/*"])
        print(f"Qwen snapshot: {directory}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-views", action="store_true")
    prepare(parser.parse_args().qwen_views)
