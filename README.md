# BuildOne

Turn one or more photos of the same rigid object, or a single video, into an editable build123d model, a USD asset with colliders, and an offline 3D preview.

## Setup

Use Linux x86-64 with a BF16-capable NVIDIA GPU and a CUDA 12.9-compatible driver. Install Bash, Git, patch, curl, and the OpenGL runtime (`libgl1` on Ubuntu). The setup script installs uv if needed, selects Python 3.11, and installs the locked dependencies. Setup and Codex both require internet access.

1. Request access to SAM 3.1 on Hugging Face. Once approved, use the same account during setup. The script reuses an existing login or prompts you to sign in through a browser or paste a read token. For unattended setup, set `HF_TOKEN`. [Request SAM 3.1 access](https://huggingface.co/facebook/sam3.1) · [Create a token](https://huggingface.co/settings/tokens)
2. Clone the repository and run setup.

   ```sh
   git clone https://github.com/klee101/buildone.git
   cd buildone
   ./setup.sh
   ```

   Setup downloads `sam3.1_multiplex.pt` (3.5 GB) to `models/sam3.1/`. It reuses your Codex login and model settings. For a first-time Codex login, follow the link in the terminal or set `OPENAI_API_KEY` before setup. Choose a model that accepts image input.

If you already have the SAM checkpoint, you can skip the Hugging Face login. After cloning and before running setup, copy `docs/example.env` to `.env` and set `BUILDONE_SAM_WEIGHTS` to the checkpoint path. You can also set `BUILDONE_AGENT_MODEL` there. Both `setup.sh` and `run.sh` read this file.

## Run

Build an asset from the three included photos.

```sh
./run.sh --input data/*.jpg --size-prompt "The object is about 17 cm tall."
```

For your own object, pass its photos or one video after `--input`. The size prompt is optional. The terminal prints the output directory. Edit `model.py` to change the CAD geometry, import `asset.usd` into Isaac Sim, or open `viewer.html` in a WebGL 2 browser.

To generate four additional reference views, install the Qwen weights and enable the option when running. Setup downloads about 32.5 GB to `models/Qwen-Image-2.1-Turbo/`. Inference needs enough GPU memory for the weights and intermediate tensors. [Qwen model files](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/main)

```sh
./setup.sh --qwen-views
./run.sh --input data/*.jpg --qwen-views
```
