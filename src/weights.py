"""Paths shared by setup and the offline inference workers."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAM_FILE = "sam3.1_multiplex.pt"
SAM_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
SAM_BYTES = 3_502_755_717
QWEN_REVISION = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"
QWEN_NAME = "Qwen-Image-2.1-Turbo"


def weight_path(variable, default):
    path = Path(os.getenv(variable) or default).expanduser()
    return (ROOT / path).resolve()


def sam_weights():
    return weight_path("BUILDONE_SAM_WEIGHTS", Path("models/sam3.1") / SAM_FILE)


def qwen_weights():
    return weight_path("BUILDONE_QWEN_WEIGHTS", Path("models") / QWEN_NAME)


def sam_ready():
    path = sam_weights()
    if not path.is_file():
        return False
    size = path.stat().st_size
    return size > 0 if os.getenv("BUILDONE_SAM_WEIGHTS") else size == SAM_BYTES


def check_qwen():
    directory = qwen_weights()
    required = ["model_index.json", "processor/processor_config.json", "processor/tokenizer.json",
                "processor/tokenizer_config.json", "processor/chat_template.jinja",
                "scheduler/scheduler_config.json"]
    for component, stem in (("transformer", "diffusion_pytorch_model"),
                            ("text_encoder", "model"), ("vae", "diffusion_pytorch_model")):
        required.append(f"{component}/config.json")
        index = directory / component / f"{stem}.safetensors.index.json"
        if index.is_file():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards:
                raise ValueError(f"Empty weight index: {index}")
            required.extend(f"{component}/{shard}" for shard in shards)
        else:
            required.append(f"{component}/{stem}.safetensors")
    for name in required:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing {path}. Run ./setup.sh --qwen-views "
                                    "or set BUILDONE_QWEN_WEIGHTS to a complete Diffusers snapshot.")


def prepare(include_qwen):
    from huggingface_hub import get_token, hf_hub_download, login, snapshot_download
    from huggingface_hub.errors import GatedRepoError

    checkpoint = sam_weights()
    if not sam_ready():
        if os.getenv("BUILDONE_SAM_WEIGHTS"):
            raise SystemExit(f"BUILDONE_SAM_WEIGHTS must point to an existing checkpoint: {checkpoint}")
        if not get_token():
            print("Request access at https://huggingface.co/facebook/sam3.1 and use a read token "
                  "from https://huggingface.co/settings/tokens after approval.", flush=True)
            if not sys.stdin.isatty():
                raise SystemExit("Set HF_TOKEN or run .venv/bin/hf auth login, then rerun setup.")
            login()
        try:
            hf_hub_download("facebook/sam3.1", SAM_FILE, revision=SAM_REVISION,
                            local_dir=checkpoint.parent, force_download=checkpoint.exists())
        except GatedRepoError as exc:
            raise SystemExit("SAM access denied. Use a read token from the account approved at "
                             "https://huggingface.co/facebook/sam3.1 and rerun ./setup.sh.") from exc
        if not sam_ready():
            raise SystemExit(f"Incomplete SAM checkpoint at {checkpoint}. Rerun ./setup.sh.")
    print(f"SAM checkpoint: {checkpoint}", flush=True)
    if include_qwen:
        directory = qwen_weights()
        if not os.getenv("BUILDONE_QWEN_WEIGHTS"):
            snapshot_download(f"Qwen/{QWEN_NAME}", revision=QWEN_REVISION,
                              local_dir=directory, ignore_patterns=["assets/*"])
        check_qwen()
        print(f"Qwen snapshot: {directory}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-views", action="store_true")
    prepare(parser.parse_args().qwen_views)
