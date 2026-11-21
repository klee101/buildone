"""One Turbo image edit creates all four orthographic references."""
import json
import os
from time import perf_counter

VIEWS = ("front", "top", "left", "right")
PROMPT = """Produce one square technical modeling reference sheet showing the same isolated object four times. Exact equal 2 by 2 grid without text, dividers or perspective. Top left: front elevation, camera on -Y. Top right: top plan from +Z, front edge at the bottom. Bottom left: left elevation from -X. Bottom right: right elevation from +X. Z is up; X is width, Y is depth. The object's front is {front}. The object is {description}. Use true orthographic projections, consistent orientation, identical physical scale, each view centered with a clear margin and plain light gray background. Preserve visible proportions, colors, handles, holes, openings, ribs and asymmetric features; complete occluded geometry conservatively. Show only the selected object."""


def generate(out):
    import torch
    from diffusers import QwenImage21Pipeline
    from PIL import Image

    target = json.loads((out / "target.json").read_text())
    model = os.getenv("BUILDONE_QWEN_WEIGHTS", "/data/hpf/models/Qwen-Image-2.1-Turbo")
    prompt = PROMPT.format(**target)
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    start = perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16,
                                               local_files_only=True).to("cuda")
    schedule = list(pipe.config.sample_sigmas or [])
    if len(schedule) != 8:
        raise ValueError("Expected the eight-step Qwen Turbo schedule")
    torch.cuda.synchronize()
    load = perf_counter() - start
    with Image.open(out / "evidence/primary-isolated.png") as source, torch.inference_mode():
        crop = source.convert("RGBA")
        start = perf_counter()
        sheet = pipe(image=crop, prompt=prompt, width=1024, height=1024, output_resolution=512,
                     true_cfg_scale=1., use_kv_cache=True,
                     generator=torch.Generator("cpu").manual_seed(42)).images[0]
        torch.cuda.synchronize()
        inference = perf_counter() - start
    if sheet.size != (1024, 1024):
        raise ValueError(f"Unexpected Qwen output size {sheet.size}")
    folder = out / "references"
    folder.mkdir(exist_ok=True)
    sheet.save(folder / "sheet.png")
    for i, name in enumerate(VIEWS):
        x, y = i % 2 * 512, i // 2 * 512
        sheet.crop((x, y, x+512, y+512)).save(folder / f"{name}.png")
    return dict(model=model, model_init_seconds=load, inference_seconds=inference,
                calls=1, steps=len(schedule), cfg=1.0, kv_cache=True, seed=42, prompt=prompt)
