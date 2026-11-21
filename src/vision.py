"""Frame selection, target identification and point-conditioned SAM mask selection."""
import json
import math
import os
from time import perf_counter

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

import jobs


IDENTIFY = """Select one physical object. Follow the target hint when present, otherwise choose the primary foreground object. For repeated objects, choose one clearly visible instance. Write target.json as JSON with "query" (short English noun phrase for SAM), "description", "front" (an identifiable physical side), "primary" ({"frame": integer ID, "point": [x,y]}), and "secondary" (null or the same instance at a useful distinct angle in another frame). Points must be strictly inside solid target material, normalized 0..1 in each original frame, measured right/down from its top left. Exclude labels and padding from coordinates."""
SAM_WEIGHTS = "/data/hpf/models/sam3.1/sam3.1_multiplex.pt"


def frames(path):
    try:
        with Image.open(path) as source:
            return {0: ImageOps.exif_transpose(source).convert("RGB")}, False
    except UnidentifiedImageError:
        import av

    # Some containers omit frame counts. Decode a counting pass in that case.
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        count = stream.frames or sum(1 for _ in container.decode(stream))
    if count < 1:
        raise ValueError("Video contains no decodable frames")
    wanted = {round(i * (count - 1) / max(1, min(count, 5) - 1)) for i in range(min(count, 5))}
    with av.open(str(path)) as container:
        sampled = {i: frame.to_image().convert("RGB")
                   for i, frame in enumerate(container.decode(video=0)) if i in wanted}
    if not sampled:
        raise ValueError("Could not decode selected frames")
    return sampled, True


def identify(source, out, hint):
    samples, video = frames(source)
    evidence = out / "evidence"
    evidence.mkdir(exist_ok=True)
    for frame_id, frame in samples.items():
        frame.save(evidence / f"frame-{frame_id}.png")
    if not video:
        sheet = evidence / "frame-0.png"
    else:
        width, height = 512, 384
        cols = min(3, len(samples))
        sheet_image = Image.new("RGB", (cols * width, math.ceil(len(samples) / cols) * (height + 26)), "white")
        draw = ImageDraw.Draw(sheet_image)
        for index, (frame_id, frame) in enumerate(samples.items()):
            x, y = (index % cols) * width, (index // cols) * (height + 26)
            tile = ImageOps.contain(frame, (width, height))
            sheet_image.paste(tile, (x + (width-tile.width)//2, y + 26 + (height-tile.height)//2))
            draw.text((x+8, y+4), f"Frame {frame_id}", fill="black")
        sheet = evidence / "contact.png"
        sheet_image.save(sheet)
    job = out / "jobs" / "identify"
    contract = IDENTIFY + f"\nFrame IDs: {list(samples)}. Target hint: {hint or '(none)'}"
    jobs.run(job, contract, "Identify the target in the attached image and write target.json.",
             [sheet], outputs=("target.json",))
    result = json.loads((job / "target.json").read_text())
    for key in ("query", "description", "front"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"Missing {key} in target identification")
    primary, secondary = result["primary"], result.get("secondary")
    if secondary and secondary["frame"] == primary["frame"]:
        raise ValueError("Secondary view duplicates the primary frame")
    for view in [primary] + ([secondary] if secondary else []):
        point = view["point"]
        if (type(view["frame"]) is not int or view["frame"] not in samples or len(point) != 2
                or not all(type(v) in (float, int) and 0 < v < 1 for v in point)):
            raise ValueError(f"Invalid selected view: {view}")
    return result


def segment(out):
    import numpy as np
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    from sam3.model.sam3_image_processor import Sam3Processor

    target = json.loads((out / "target.json").read_text())
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    start = perf_counter()
    checkpoint = os.getenv("BUILDONE_SAM_WEIGHTS", SAM_WEIGHTS)
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path=checkpoint, use_fa3=False, use_rope_real=False,
        compile=False, warm_up=False, async_loading_frames=False)
    # The point disambiguates proposals. Keep even low-confidence matches for unusual parts.
    processor = Sam3Processor(predictor.model.detector, confidence_threshold=0.0)
    torch.cuda.synchronize()
    result = dict(checkpoint=checkpoint, model_init_seconds=perf_counter()-start,
                  inference_seconds=0.0, views=[])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for role in ("primary", "secondary"):
            view = target.get(role)
            if not view:
                continue
            pic = Image.open(out / "evidence" / f"frame-{view['frame']}.png").convert("RGB")
            x, y = (int(c*s) for c, s in zip(view["point"], pic.size))
            start = perf_counter()
            state = processor.set_text_prompt(target["query"], processor.set_image(pic))
            scores = state["scores"].masked_fill(~state["masks"][:, 0, y, x], -float("inf"))
            score = scores.max().item() if scores.numel() else -float("inf")
            mask = state["masks"][scores.argmax(), 0].cpu().numpy() if math.isfinite(score) else None
            torch.cuda.synchronize()
            result["inference_seconds"] += perf_counter() - start
            if mask is None:
                if role == "secondary":
                    result["secondary_skipped"] = "No SAM mask contained the selected point"
                    continue
                raise ValueError(f"No SAM mask contains primary point {view['point']} for {target['query']!r}")
            matte = Image.fromarray(np.uint8(mask)*255)
            x0, y0, x1, y1 = matte.getbbox()
            pad = max(3, round(.035*max(x1-x0, y1-y0)))
            box = (max(0, x0-pad), max(0, y0-pad), min(pic.width, x1+pad), min(pic.height, y1+pad))
            crop, alpha = pic.crop(box), matte.crop(box)
            dest = out / "evidence"
            crop.save(dest / f"{role}-source.png")
            alpha.save(dest / f"{role}-mask.png")
            isolated = Image.composite(crop, Image.new("RGB", crop.size, (222, 222, 222)), alpha)
            isolated.save(dest / f"{role}-isolated.png")
            result["views"].append(dict(role=role, frame=view["frame"], score=score, box_xyxy=box,
                                        point_px=[x, y], mask_area_px=int(mask.sum())))
            del state, scores
    return result
