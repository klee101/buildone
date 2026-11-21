# BuildOne

Build one editable CAD model and an Isaac Sim USD asset from a still image or a video.

## Setup

On Linux x86-64 with Python 3.11, an NVIDIA CUDA-compatible driver, Git, `patch`,
`uv`, and an authenticated Codex CLI, run:

```sh
./setup.sh
```

Setup initializes the pinned build123d, Diffusers, and SAM source submodules,
installs the locked Python dependencies into `.venv/`, and applies the SAM 3.1
merged-checkpoint fix to the installed package.

The default weight paths are:

| Model | Path | Override |
|---|---|---|
| SAM 3.1 | `/data/hpf/models/sam3.1/sam3.1_multiplex.pt` | `BUILDONE_SAM_WEIGHTS` |
| Qwen-Image-2.1-Turbo | `/data/hpf/models/Qwen-Image-2.1-Turbo` | `BUILDONE_QWEN_WEIGHTS` |

## Run

```sh
.venv/bin/python -B src/run.py --input /path/to/component.jpg
```

The run prints its directory under `output/`. It contains `asset.usd`, `model.py`,
four modeling references, and `timing.json`. Pass `--out output/my-part` to set a
specific run directory. Use the same `--out` and input with `--resume` to continue
an interrupted run.
