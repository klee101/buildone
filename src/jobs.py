"""Prepare local evidence and consume the files from a Codex job."""
import json
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys


RUNNER = Path(__file__).with_name("agentrunner.py")
RULES = """## Job rules

Work inside this directory and use the supplied inputs. Use python3 and its standard library
if needed to write outputs. The pipeline executes CAD builds. In REPORT.md, briefly
state what you did, how you checked it, and any doubts.
"""


def run(job, contract, prompt, images, *, outputs, turn=None, optional_outputs=()):
    logs = job / ".agent"
    if turn:
        logs /= turn
    result_path = logs / "result.json"
    expected = [job / name for name in outputs]
    if result_path.is_file() and all(path.is_file() for path in expected):
        if json.loads(result_path.read_text())["status"] == "completed":
            return

    logs.mkdir(parents=True, exist_ok=True)
    (job / "AGENTS.md").write_text(contract.rstrip() + "\n\n" + RULES)
    for image in images:
        copyfile(image, job / image.name)
    for path in (result_path, *expected, *(job / name for name in optional_outputs)):
        path.unlink(missing_ok=True)

    request = dict(job_dir=str(job), prompt=prompt, images=[p.name for p in images], turn=turn)
    python = os.getenv("BUILDONE_RUNNER_PYTHON") or sys.executable
    with (logs / "runner.log").open("w") as log:
        process = subprocess.run(
            [python, "-B", str(RUNNER)], input=json.dumps(request), text=True,
            stdout=log, stderr=subprocess.STDOUT, timeout=3600,
        )
    if process.returncode:
        raise RuntimeError(f"Codex job {job.name} failed. See {logs / 'runner.log'}")
    result = json.loads(result_path.read_text())
    if result["status"] != "completed":
        raise RuntimeError(f"Codex job {job.name} {result['status']}: {result['error']}")
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Codex job {job.name} did not write {', '.join(missing)}")
