# BuildOne

BuildOne turns a photo or video of one rigid object into editable CAD and an Isaac Sim asset.

## The stages

| Stage | What the agent supplies | What the code produces |
| --- | --- | --- |
| Target | Object label, frame, and point | SAM 3.1 mask and isolated crop |
| References | | Qwen front, top, left, and right views |
| CAD | build123d source, with up to one revision | Mesh and orthographic preview |
| Export | | Self-contained USD with colliders and rigid-body properties |

## What it needs

Linux x86-64, Python 3.11, an NVIDIA GPU with a compatible driver, Git, `patch`, `uv`, and the Codex CLI for login. The Python SDK and its runtime are installed by setup.

| Model path override | Default |
| --- | --- |
| `BUILDONE_SAM_WEIGHTS` | `/data/hpf/models/sam3.1/sam3.1_multiplex.pt` |
| `BUILDONE_QWEN_WEIGHTS` | `/data/hpf/models/Qwen-Image-2.1-Turbo` |

## Install

```sh
./setup.sh
CODEX_HOME="${BUILDONE_CODEX_HOME:-$HOME/.buildone/codex}" codex login
```

## Quick start

```sh
.venv/bin/python -B src/run.py --input /path/to/component.jpg
```

Each run writes `asset.usd`, `model.py`, four reference views, and `timing.json` under `output/`. Codex jobs keep their contracts, outputs, and turn records under `jobs/` in the run. Pass a measured dimension such as `--width-mm 120` to set the scale.

Run `BUILDONE_ENV_FILE=path/to/file source src/env.sh` to load the model, effort, and runner settings from a file. See [docs/example.env](docs/example.env).
