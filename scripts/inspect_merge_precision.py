"""Diagnose CPU vs MPS BF16 merging using only LoRA target matrices."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--export-dir", default="outputs/qwen2vl_merged_cpu")
parser.add_argument("--output", default="logs/merge_precision.json")
args = parser.parse_args()
base_dir = root / "models/Qwen2-VL-2B-Instruct"
export_dir = root / args.export_dir
adapter_dir = root / "outputs/qwen2vl_lora_mps"
base_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
export_map = json.loads((export_dir / "model.safetensors.index.json").read_text())["weight_map"]
adapter = load_file(str(adapter_dir / "adapter_model.safetensors"))
config = json.loads((adapter_dir / "adapter_config.json").read_text())
scale = config["lora_alpha"] / config["r"]
rows = []
with torch.inference_mode():
    for name, a in adapter.items():
        if not name.endswith(".lora_A.weight"):
            continue
        b = adapter[name.replace(".lora_A.weight", ".lora_B.weight")]
        target = name.removeprefix("base_model.model.").replace("model.language_model.", "model.").replace(".lora_A.weight", ".weight")
        with safe_open(str(base_dir / base_map[target]), framework="pt", device="cpu") as f:
            base = f.get_tensor(target)
        with safe_open(str(export_dir / export_map[target]), framework="pt", device="cpu") as f:
            exported = f.get_tensor(target)
        cpu_merged = base.clone()
        cpu_merged.add_((b @ a) * scale)
        mps_merged = base.to("mps")
        mps_merged.add_((b.to("mps") @ a.to("mps")) * scale)
        mps_merged = mps_merged.cpu()
        rows.append({"target": target,
                     "export_vs_cpu_unequal": int((exported != cpu_merged).sum()),
                     "export_vs_mps_unequal": int((exported != mps_merged).sum()),
                     "cpu_vs_mps_max_abs": float((cpu_merged.float() - mps_merged.float()).abs().max())})
report = {"target_matrices": len(rows),
          "export_vs_cpu_unequal_elements": sum(r["export_vs_cpu_unequal"] for r in rows),
          "export_vs_mps_unequal_elements": sum(r["export_vs_mps_unequal"] for r in rows),
          "cpu_vs_mps_max_abs": max(r["cpu_vs_mps_max_abs"] for r in rows), "details": rows}
(root / args.output).write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
