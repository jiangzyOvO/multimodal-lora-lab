"""Run identical deterministic image prompts before/after SFT through LLaMA-Factory."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import torch
from llamafactory.chat import ChatModel

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/Qwen2-VL-2B-Instruct")
parser.add_argument("--adapter")
parser.add_argument("--output", required=True)
args = parser.parse_args()
torch.manual_seed(42)
config = {"model_name_or_path": str(root / args.model),
          "template": "qwen2_vl", "finetuning_type": "lora", "infer_dtype": "bfloat16",
          "infer_backend": "huggingface", "flash_attn": "disabled", "trust_remote_code": False,
          "image_max_pixels": 50176, "image_min_pixels": 3136, "do_sample": False,
          "max_new_tokens": 64}
if args.adapter:
    config["adapter_name_or_path"] = str(root / args.adapter)
model = ChatModel(config)
# Transformers can fill GenerationConfig defaults from the model's saved config.
# Make both layers explicitly greedy, including after export/reload.
model.engine.model.generation_config.do_sample = False
model.engine.model.generation_config.temperature = None
model.engine.model.generation_config.top_p = None
model.engine.model.generation_config.top_k = None
print("INFERENCE_MODEL: " + json.dumps({"devices": sorted({str(p.device) for p in model.engine.model.parameters()}),
                                       "dtypes": sorted({str(p.dtype) for p in model.engine.model.parameters()}),
                                       "do_sample": model.engine.model.generation_config.do_sample}), flush=True)
cases = [("seen_1", "1.jpg", "<image>他们是谁？"),
         ("seen_2", "2.jpg", "<image>他为什么在地上？"),
         ("holdout_3", "3.jpg", "<image>请描述这张图片")]
rows = []
for name, filename, prompt in cases:
    start = time.monotonic()
    responses = model.chat([{"role": "user", "content": prompt}],
                           images=[str(root / "data/mllm_demo_data" / filename)], do_sample=False)
    torch.mps.synchronize()
    row = {"case": name, "image": filename, "prompt": prompt, "response": responses[0].response_text,
           "seconds": round(time.monotonic() - start, 3),
           "mps_allocated_gib": round(torch.mps.current_allocated_memory() / 2**30, 3)}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    (root / args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
