"""Execute and record one Codex SDK turn in an isolated job directory."""
from enum import Enum
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from openai_codex import ApprovalMode, Codex, CodexConfig, LocalImageInput, Sandbox, TextInput


def codex_env():
    home = os.getenv("BUILDONE_CODEX_HOME") or Path.home() / ".buildone" / "codex"
    env = {"CODEX_HOME": str(Path(home).expanduser().resolve())}
    if directory := os.getenv("BUILDONE_BWRAP_DIR"):
        bwrap = Path(directory).expanduser().resolve() / "bwrap"
        if not bwrap.is_file():
            raise FileNotFoundError(f"Missing sandbox executable: {bwrap}")
        system_bwrap = Path("/usr/bin/bwrap")
        if system_bwrap.is_file() and bwrap.read_bytes() != system_bwrap.read_bytes():
            print(f"Warning: {bwrap} differs from {system_bwrap}", file=sys.stderr)
        env["PATH"] = os.pathsep.join((str(bwrap.parent), os.getenv("PATH", "")))
    return env


def dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    return value


def run_job(job_dir, prompt, images, turn):
    job = Path(job_dir).resolve()
    if not (job / ".git").is_dir():
        subprocess.run(["git", "init", "-q", str(job)], check=True)
    logs = job / ".agent"
    if turn:
        logs /= turn
    logs.mkdir(parents=True, exist_ok=True)

    model = os.getenv("BUILDONE_AGENT_MODEL", "gpt-6-astra")
    effort = os.getenv("BUILDONE_AGENT_EFFORT", "high")
    inputs = [TextInput(text=prompt)]
    inputs.extend(LocalImageInput(path=str(job / image)) for image in images)
    start = perf_counter()
    with Codex(CodexConfig(cwd=str(job), env=codex_env())) as codex:
        thread = codex.thread_start(
            cwd=str(job), model=model, sandbox=Sandbox.workspace_write,
            approval_mode=ApprovalMode.deny_all,
        )
        result = thread.run(inputs, effort=effort)
    record = dict(
        thread_id=thread.id, turn_id=result.id, status=dump(result.status),
        error=dump(result.error), model=model, effort=effort,
        sdk=version("openai-codex"), codex=version("openai-codex-cli-bin"),
        wall_s=round(perf_counter() - start, 2), duration_ms=result.duration_ms,
        final_response=result.final_response, usage=dump(result.usage),
    )
    with (logs / "items.jsonl").open("w") as stream:
        for item in result.items:
            stream.write(json.dumps(dump(item)) + "\n")
    temp = logs / "result.json.tmp"
    temp.write_text(json.dumps(record, indent=2) + "\n")
    temp.replace(logs / "result.json")


if __name__ == "__main__":
    run_job(**json.load(sys.stdin))
