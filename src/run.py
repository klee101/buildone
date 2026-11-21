"""Build one CAD and USD asset from an image or video.

Example: .venv/bin/python -B src/run.py --input /data/photo.jpg
Override the weight paths with BUILDONE_SAM_WEIGHTS and BUILDONE_QWEN_WEIGHTS.
The default image-capable agent runtime is the Codex CLI.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp
from time import perf_counter, sleep
import traceback

sys.dont_write_bytecode = True
ENTRY = Path(__file__).resolve()
ROOT = ENTRY.parents[1]
OUTPUT = ROOT / "output"
GPU_WORKERS = {"sam", "qwen"}


def write(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def await_file(path):
    while not path.is_file():
        sleep(.15)


class Agents:
    def __init__(self, out, hosted):
        self.folder = out / "agents"
        self.hosted = hosted

    def ask(self, name, prompt, images, *, suffix, session=None):
        folder = self.folder / name
        folder.mkdir(parents=True, exist_ok=True)
        answer = folder / ("response" + suffix)
        if not (folder/"request.json").exists():
            write(folder/"request.json", dict(prompt=prompt, images=list(map(str, images)),
                                              session=session or name, response=str(answer)))
        if self.hosted and not answer.exists():
            print(f"Agent request: {folder/'request.json'}", flush=True)
            await_file(answer)
        elif not self.hosted and not answer.exists():
            cmd = ["codex", "--no-daemon", "--ask-for-approval", "never", "--sandbox", "read-only",
                   "--disable", "apps", "--disable", "plugins", "--disable", "shell_tool",
                   "--enable", "skip_host_skill_discovery", "exec"]
            if session:
                cmd.append("resume")
            elif name == "identify":
                cmd.append("--ephemeral")
            cmd.extend(["--skip-git-repo-check", "--json", "-o", str(answer)])
            for image in images:
                cmd.extend(["--image", str(image)])
            if session:
                cmd.append(session)
            cmd.append("-")
            env = os.environ.copy()
            for key in ("CODEX_THREAD_ID", "CODEX_SESSION_ID"):
                env.pop(key, None)
            with (folder/"events.jsonl").open("w") as log, (folder/"stderr.log").open("w") as err:
                subprocess.run(cmd, cwd=self.folder.parent, env=env, input=prompt, text=True,
                               stdout=log, stderr=err, check=True)
        if not self.hosted and name == "model":
            events = map(json.loads, (folder/"events.jsonl").read_text().splitlines())
            session = next(e["thread_id"] for e in events if e.get("type") == "thread.started")
        output = answer.read_text().strip()
        if output.startswith("```") and output.endswith("```"):
            output = output.split("\n", 1)[1].rsplit("```", 1)[0].rstrip()
        return output + "\n", session or name


def worker(args):
    out, name = args.out, args.worker
    folder = out / "cad" / f"v{args.version}" if name == "cad" else out
    dest = folder / (name + ".json")
    start = perf_counter()
    try:
        if name == "sam":
            from vision import segment
            result = segment(out)
        elif name == "qwen":
            from reference import generate
            result = generate(out)
        else:
            from cad import build_version
            result = build_version(folder, read(out/"settings.json")["known_mm"])
        result.update(worker_seconds=perf_counter()-start)
        write(dest, result)
    except Exception:
        write(dest, dict(usable=False, error=traceback.format_exc(), worker_seconds=perf_counter()-start))
        raise


def run(args, known):
    from vision import identify
    out = args.out
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    settings = dict(input=str(args.input), known_mm=known, static=args.static, mass_kg=args.mass_kg)
    if args.hint:
        settings["hint"] = args.hint
    start = perf_counter()
    agents = Agents(out, args.host_agents)
    timing = dict(started_at=datetime.now(timezone.utc).isoformat(), input=str(args.input),
                  complete=False, stage_seconds={}, scale_source="estimated" if not known else
                  ("supplied" if len(known)==3 else "one supplied axis; other dimensions estimated"),
                  known_mm=known)
    if args.resume:
        if read(out/"settings.json") != settings:
            raise ValueError("Resume requires the original input, dimensions, hint and physics settings")
        timing = read(out/"timing.json")
        if timing["complete"]:
            return
        start -= (datetime.now(timezone.utc)-datetime.fromisoformat(timing["started_at"])).total_seconds()
        timing.pop("error", None)
    else:
        write(out/"settings.json", settings)
        write(out/"timing.json", timing)

    @contextmanager
    def stage(name):
        print(name, flush=True)
        begin = perf_counter()
        marks = timing.setdefault("stage_started_at", {})
        now = datetime.now(timezone.utc)
        began_at = marks.setdefault(name, now.isoformat())
        # Include time spent waiting for the same agent if the coordinator resumes.
        begin -= (now-datetime.fromisoformat(began_at)).total_seconds()
        write(out/"timing.json", timing)
        try:
            yield
        finally:
            timing["stage_seconds"][name] = perf_counter()-begin
            timing["end_to_end_seconds"] = perf_counter()-start
            write(out/"timing.json", timing)

    def invoke(name, folder, version=0):
        python = os.getenv(f"BUILDONE_{name.upper()}_PYTHON", sys.executable)
        cmd = [python, "-B", "-u", str(ENTRY), "--out", str(out), "--worker", name]
        if name == "cad":
            cmd.extend(["--version", str(version)])
        dest = folder/(name+".json")
        if args.host_gpu and name in GPU_WORKERS:
            request = out/(name+"-request.json")
            if not request.exists():
                write(request, dict(argv=cmd, result=str(dest)))
            print(f"GPU request: {out/(name+'-request.json')}", flush=True)
            await_file(dest)
        else:
            log = folder/(name+".log")
            with log.open("w") as output:
                completed = subprocess.run(cmd, cwd=folder, stdout=output, stderr=subprocess.STDOUT)
            if not dest.is_file():
                error = f"{name} process exited {completed.returncode}: {log.read_text()[-6000:]}"
                if name == "cad":
                    return dict(usable=False, error=error)
                raise RuntimeError(error)
        result = read(dest)
        if name != "cad" and "error" in result:
            raise RuntimeError(result["error"])
        return result

    try:
        if args.resume and (out/"target.json").exists():
            target = read(out/"target.json")
        else:
            with stage("identification"):
                target = identify(args.input, out, args.hint, agents.ask)
                write(out/"target.json", target)
        for name in ("sam", "qwen"):
            if args.resume and (out/(name+".json")).exists():
                result = read(out/(name+".json"))
                if "error" in result:
                    raise RuntimeError(result["error"])
                timing[name] = result
                request = out/(name+"-request.json")
                duration = ((out/(name+".json")).stat().st_mtime-request.stat().st_mtime
                            if request.exists() else result["worker_seconds"])
                timing["stage_seconds"].setdefault(name, duration)
            else:
                with stage(name):
                    timing[name] = invoke(name, out)
        sam = timing["sam"]
        if args.resume and timing.get("cad", {}).get("mesh_cache"):
            cache = out/timing["cad"]["mesh_cache"]
        else:
            with stage("cad"):
                from cad import modeling
                cad = timing["cad"] = dict(build_count=0, attempts=[])
                cache = modeling(out, target, sam, known, agents.ask,
                                 lambda folder, v: invoke("cad", folder, v), cad)
        with stage("usd"):
            from usd import export
            temp = out/"asset.partial.usd"
            timing["physics"] = export(cache, temp, args.static, args.mass_kg)
            temp.replace(out/"asset.usd")
        timing.update(complete=True, finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        timing["error"] = str(exc)
        raise
    finally:
        timing["end_to_end_seconds"] = perf_counter()-start
        timing["outside_stages_seconds"] = max(0, timing["end_to_end_seconds"] -
                                               sum(timing["stage_seconds"].values()))
        write(out/"timing.json", timing)
    print(f"Exported {out/'asset.usd'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Source image or video")
    parser.add_argument("--out", type=Path, help="Run directory (default: a new directory under output/)")
    parser.add_argument("--hint", help="Preferred target")
    for axis in ("width", "depth", "height"):
        parser.add_argument(f"--{axis}-mm", type=float)
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument("--static", action="store_true", help="Author a static compound collider")
    parser.add_argument("--resume", action="store_true", help="Continue this run using its completed stages")
    parser.add_argument("--host-agents", action="store_true",
                        help="Agent host consumes agents/*/request.json and atomically writes each response")
    parser.add_argument("--host-gpu", action="store_true",
                        help="GPU host executes argv in sam-request.json and qwen-request.json")
    parser.add_argument("--worker", choices=["sam", "qwen", "cad"], help=argparse.SUPPRESS)
    parser.add_argument("--version", type=int, choices=[0, 1], default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.out is not None:
        args.out = args.out.resolve()
        if args.out.is_relative_to(ROOT) and not args.out.is_relative_to(OUTPUT):
            parser.error("Use output/ for generated files inside BuildOne")
    elif args.worker or args.resume:
        parser.error("--out is required when resuming or launching a worker")
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="8")
    if args.worker:
        worker(args)
        return
    if args.input is None or not args.input.is_file():
        parser.error("Provide a readable image or video with --input")
    args.input = args.input.resolve()
    known = {k: getattr(args, k+"_mm") for k in ("width", "depth", "height")
             if getattr(args, k+"_mm") is not None}
    values = list(known.values()) + ([args.mass_kg] if args.mass_kg is not None else [])
    if len(known) == 2 or any(not math.isfinite(v) or v <= 0 for v in values):
        parser.error("Supply one known axis or all three. Dimensions and mass must be positive and finite.")
    if args.out is None:
        OUTPUT.mkdir(exist_ok=True)
        args.out = Path(mkdtemp(prefix=f"{args.input.stem}-", dir=OUTPUT))
    else:
        args.out.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {args.out}", flush=True)
    lock = os.open(args.out, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, known)
    finally:
        os.close(lock)


if __name__ == "__main__":
    main()
