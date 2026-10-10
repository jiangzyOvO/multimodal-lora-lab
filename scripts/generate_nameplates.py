"""Generate a deterministic, image-disjoint and template-disjoint OCR/SFT dataset.

Images and labels come from the same explicit rendering specification. No model is
used to invent annotations. A separate validator checks the emitted files.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from functools import lru_cache
from pathlib import Path

import PIL
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261009
SIZE = (672, 448)
FIELDS = ["型号", "电压", "功率"]
PROMPT = (
    '<image>读取设备铭牌，只输出一个JSON对象，键必须为“型号”“电压”“功率”，不要解释或代码块。'
    '型号按图片原样抄录；电压只保留数字和V，去掉AC/DC及空格；功率只保留数字和W，去掉空格。'
    '字段未出现、值为空或标为--、N/A时填写null。不要根据其他字段推测缺失值。'
)
FONTS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
]
# Geometry and treatment together define a template. No template crosses splits.
TEMPLATES = {
    "T01": ("train", "rows", [(44, 94, 628, 166), (44, 180, 628, 252), (44, 266, 628, 338)]),
    "T02": ("train", "grid", [(60, 110, 612, 182), (60, 182, 612, 254), (60, 254, 612, 326)]),
    "T03": ("train", "columns", [(44, 108, 228, 340), (244, 108, 428, 340), (444, 108, 628, 340)]),
    "T04": ("train", "topwide", [(44, 94, 628, 166), (44, 188, 326, 338), (346, 188, 628, 338)]),
    "T05": ("train", "leftlarge", [(44, 100, 308, 338), (328, 100, 628, 209), (328, 229, 628, 338)]),
    "T06": ("train", "band", [(120, 94, 628, 166), (120, 180, 628, 252), (120, 266, 628, 338)]),
    "T07": ("train", "narrow", [(82, 114, 590, 186), (82, 192, 590, 264), (82, 270, 590, 342)]),
    "T08": ("train", "dark", [(44, 94, 628, 166), (44, 180, 628, 252), (44, 266, 628, 338)]),
    "V01": ("val", "bottomwide", [(44, 100, 326, 208), (346, 100, 628, 208), (44, 230, 628, 338)]),
    "V02": ("val", "stagger", [(44, 94, 500, 166), (108, 180, 564, 252), (172, 266, 628, 338)]),
    "E01": ("test", "rightlarge", [(44, 100, 344, 209), (44, 229, 344, 338), (364, 100, 628, 338)]),
    "E02": ("test", "offsetcolumns", [(44, 94, 228, 302), (244, 130, 428, 338), (444, 112, 628, 320)]),
}
LABELS = {
    "型号": ["型号", "产品型号", "Model"],
    "电压": ["额定电压", "输入电压", "Voltage"],
    "功率": ["额定功率", "输入功率", "Power"],
}
VOLTAGES = [12, 24, 36, 48, 110, 220, 230, 380]
POWERS = [5, 12, 18, 25, 40, 60, 75, 100, 150, 200, 350, 500, 750, 1000, 1200, 1500, 1800, 2200, 3000, 5500]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@lru_cache(maxsize=256)
def font(path, size):
    return ImageFont.truetype(path, size)


def put_text(draw, text, box, font_path, size, color, annotations, kind, field=None, centered=False):
    if not text:
        return
    x1, y1, x2, y2 = box
    while size >= 14:
        face = font(font_path, size)
        bounds = draw.textbbox((0, 0), text, font=face, anchor="lt")
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        if width <= x2 - x1 and height <= y2 - y1:
            break
        size -= 1
    else:
        raise ValueError(f"Text does not fit: {text}, {box}")
    x = (x1 + (x2 - x1 - width) / 2 if centered else x1) - bounds[0]
    y = y1 + (y2 - y1 - height) / 2 - bounds[1]
    draw.text((x, y), text, font=face, fill=color, anchor="lt")
    actual_box = draw.textbbox((x, y), text, font=face, anchor="lt")
    assert x1 <= actual_box[0] and y1 <= actual_box[1] and actual_box[2] <= x2 and actual_box[3] <= y2
    annotations.append({"kind": kind, "field": field, "text": text, "bbox": list(actual_box), "font_size": size})


def render(spec, path):
    rng = random.Random(spec["render_seed"])
    _, style, boxes = TEMPLATES[spec["template"]]
    dark = style == "dark"
    background = (35, 44, 57) if dark else rng.choice([(244, 243, 236), (235, 239, 242), (247, 248, 249)])
    ink = (248, 249, 252) if dark else (27, 36, 44)
    accent = (109, 197, 212) if dark else rng.choice([(34, 86, 105), (36, 84, 61), (96, 63, 45)])
    image = Image.new("RGB", SIZE, background)
    draw = ImageDraw.Draw(image)
    annotations = []
    draw.rounded_rectangle((18, 18, 654, 430), radius=12, outline=accent, width=3)
    for x, y in [(31, 31), (641, 31), (31, 417), (641, 417)]:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=accent, width=2)
    if style == "band":
        draw.rectangle((44, 92, 95, 340), fill=accent)
    heading = rng.choice(["设备铭牌", "产品参数", "技术参数", "EQUIPMENT DATA"])
    put_text(draw, heading, (46, 39, 626, 80), spec["font"], 29, ink, annotations, "heading", centered=True)
    draw.line((44, 86, 628, 86), fill=accent, width=2)
    for field, box in zip(spec["field_order"], boxes):
        if spec["missing_field"] == field and spec["missing_mode"] == "omit":
            continue
        x1, y1, x2, y2 = box
        fill = (45, 57, 70) if dark else (255, 255, 255)
        draw.rounded_rectangle(box, radius=0 if style == "grid" else 6, fill=fill, outline=accent, width=2)
        horizontal = x2 - x1 >= 400
        if horizontal:
            divider = x1 + int((x2 - x1) * 0.44)
            label_box = (x1 + 15, y1 + 12, divider - 12, y2 - 12)
            value_box = (divider + 15, y1 + 10, x2 - 13, y2 - 10)
            if style == "grid":
                draw.line((divider, y1, divider, y2), fill=accent, width=1)
        else:
            label_box = (x1 + 10, y1 + 13, x2 - 10, y1 + (y2 - y1) * 0.40)
            value_box = (x1 + 10, y1 + (y2 - y1) * 0.45, x2 - 10, y2 - 16)
        put_text(draw, spec["field_labels"][field], label_box, spec["font"], 22, ink, annotations, "label", field, not horizontal)
        put_text(draw, spec["display_values"][field], value_box, spec["font"], 33, ink, annotations, "value", field, not horizontal)
    for index, text in enumerate(spec["distractors"]):
        put_text(draw, text, (48, 359 + index * 26, 626, 383 + index * 26), spec["font"], 18, ink, annotations, "distractor")
    image.save(path, format="PNG")
    return annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/nameplates_v1")
    parser.add_argument("--register", action="store_true", help="Add entries to data/dataset_info.json without removing existing entries.")
    args = parser.parse_args()
    destination = (ROOT / args.output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {destination}. Choose a new --output-dir for regeneration.")
    for path in FONTS:
        assert Path(path).is_file(), path
    destination.mkdir(parents=True)
    (destination / "images").mkdir()
    rng = random.Random(SEED)
    prefixes = ["AB", "CX", "DK", "FM", "GH", "JN", "KR", "LT", "MX", "NP", "QS", "RT", "SV", "TZ", "UX", "VX", "WY", "ZK"]
    models = rng.sample([f"{prefix}-{number}" for prefix in prefixes for number in range(100, 1000)], 600)
    metadata, split_summary = [], {}
    for split, count in [("train", 400), ("val", 100), ("test", 100)]:
        template_ids = [key for key, value in TEMPLATES.items() if value[0] == split]
        schedules = {
            "missing": [None] * (count * 7 // 10) + [f for f in FIELDS for _ in range(count // 10)],
            "template": [template_ids[i % len(template_ids)] for i in range(count)],
            "font": [FONTS[i % len(FONTS)] for i in range(count)],
            "voltage": [VOLTAGES[i % len(VOLTAGES)] for i in range(count)],
            "power": [POWERS[i % len(POWERS)] for i in range(count)],
        }
        for values in schedules.values():
            rng.shuffle(values)
        records = []
        for index in range(count):
            sample_id = f"{rng.getrandbits(64):016x}"
            model = models[len(metadata)]
            voltage, power = schedules["voltage"][index], schedules["power"][index]
            missing = schedules["missing"][index]
            mode = rng.choice(["omit", "blank", "marker"]) if missing else None
            truth = {"型号": model, "电压": f"{voltage}V", "功率": f"{power}W"}
            prefix = "AC" if voltage >= 110 else "DC"
            display = {"型号": model, "电压": rng.choice([f"{voltage}V", f"{voltage} V", f"{prefix} {voltage} V"]),
                       "功率": rng.choice([f"{power}W", f"{power} W"])}
            if missing:
                truth[missing] = None
                display[missing] = rng.choice(["--", "N/A"]) if mode == "marker" else ""
            order = FIELDS.copy()
            rng.shuffle(order)
            frequency = rng.choice(["50 Hz", "60 Hz", "50/60 Hz"])
            spec = {"id": sample_id, "split": split, "template": schedules["template"][index],
                    "model_group": model, "font": schedules["font"][index],
                    "render_seed": rng.getrandbits(32), "field_order": order,
                    "field_labels": {f: rng.choice(LABELS[f]) for f in FIELDS},
                    "display_values": display, "answer": truth, "missing_field": missing, "missing_mode": mode,
                    "distractors": [f"S/N: {rng.randrange(10000000, 99999999)}     Frequency: {frequency}",
                                    f"Date: 2025-{rng.randrange(1, 13):02d}     Weight: {rng.randrange(1, 30)}.{rng.randrange(10)} kg"]}
            image_path = destination / "images" / f"{sample_id}.png"
            spec["rendered_text"] = render(spec, image_path)
            spec["image_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
            spec["image"] = image_path.relative_to(ROOT / "data").as_posix()
            metadata.append(spec)
            records.append({"messages": [{"role": "user", "content": PROMPT},
                                         {"role": "assistant", "content": json.dumps(truth, ensure_ascii=False, separators=(",", ":"))}],
                            "images": [spec["image"]]})
        dump(destination / f"{split}.json", records)
        group = [row for row in metadata if row["split"] == split]
        split_summary[split] = {"count": count, "templates": dict(Counter(row["template"] for row in group)),
                                "missing_fields": dict(Counter(row["missing_field"] or "none" for row in group)),
                                "missing_modes": dict(Counter(row["missing_mode"] or "none" for row in group)),
                                "fonts": dict(Counter(Path(row["font"]).stem for row in group))}
        print(f"Generated {split}: {count}", flush=True)
    (destination / "metadata.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata))
    manifest = {"kind": "synthetic_equipment_nameplates", "version": 1, "seed": SEED, "image_size": SIZE,
                "pillow": PIL.__version__, "prompt": PROMPT, "splits": split_summary,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "font_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in FONTS},
                "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob("*.json*")}}
    dump(destination / "manifest.json", manifest)
    if args.register:
        registry_path = ROOT / "data/dataset_info.json"
        registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
        for split in split_summary:
            name = f"nameplate_{split}"
            if name in registry:
                raise ValueError(f"Dataset registration already exists: {name}")
            registry[name] = {"file_name": (destination / f"{split}.json").relative_to(ROOT / "data").as_posix(),
                              "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
                              "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
        dump(registry_path, registry)
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
