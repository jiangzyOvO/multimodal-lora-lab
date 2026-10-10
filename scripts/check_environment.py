"""Verify the actual Mac training backend before downloading model weights."""
import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from accelerate import Accelerator

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
args = parser.parse_args()
torch.manual_seed(42)
result = {
    "python": sys.version,
    "macos": platform.mac_ver()[0],
    "architecture": platform.machine(),
    "versions": {p: version(p) for p in ["torch", "torchvision", "transformers", "accelerate", "peft", "llamafactory"]},
    "mps_built": torch.backends.mps.is_built(),
    "mps_available": torch.backends.mps.is_available(),
    "mixed_precision": args.precision,
}
assert result["mps_available"], "MPS is unavailable; do not silently train on CPU."
accelerator = Accelerator(mixed_precision=args.precision)
linear = torch.nn.Linear(8, 4)
optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3)
linear, optimizer = accelerator.prepare(linear, optimizer)
before = linear.weight.detach().clone()
x = torch.randn(2, 8, device=accelerator.device)
with accelerator.autocast():
    loss = linear(x).square().mean()
accelerator.backward(loss)
optimizer.step()
torch.mps.synchronize()
result.update(device=str(accelerator.device), loss=loss.item(), updated=bool((before != linear.weight).any().item()))
assert result["updated"] and torch.isfinite(loss), "Basic training did not update parameters."
print(json.dumps(result, indent=2, ensure_ascii=False))
(root / "logs/environment.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
