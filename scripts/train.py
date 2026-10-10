"""Use LLaMA-Factory's trainer with a small callback auditing real LoRA updates."""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import torch
import yaml
from transformers import TrainerCallback
from llamafactory.train.tuner import run_exp

root = Path(__file__).resolve().parents[1]
os.chdir(root)
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/train_mps.yaml")
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--output-dir", help="Use a fresh directory for a new experiment.")
args = parser.parse_args()
config = yaml.safe_load((root / args.config).read_text())
config["logging_nan_inf_filter"] = False
if args.smoke:
    config.update(max_steps=1, warmup_steps=0, do_eval=False, eval_strategy="no",
                  save_steps=1, output_dir="outputs/smoke_mps")
if args.output_dir:
    config["output_dir"] = args.output_dir


class AuditCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        assert self.initial, "No trainable parameters"
        assert all("lora_" in n and ".visual." not in n for n in self.initial), "Unexpected trainable parameters"
        self.report = {"trainable_parameters": sum(p.numel() for p in self.initial.values()),
                       "trainable_tensors": len(self.initial), "all_trainable_are_language_lora": True,
                       "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                       "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
                       "precision": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
                       "steps": []}
        self.save(args)

    def save(self, args):
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_audit.json").write_text(json.dumps(self.report, indent=2))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            assert math.isfinite(logs["loss"]), f"Non-finite loss: {logs}"

    def on_step_end(self, args, state, control, model=None, **kwargs):
        torch.mps.synchronize()
        self.report["steps"].append({"step": state.global_step,
                                     "mps_allocated_gib": round(torch.mps.current_allocated_memory() / 2**30, 3),
                                     "mps_driver_gib": round(torch.mps.driver_allocated_memory() / 2**30, 3)})
        self.save(args)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        changes = {}
        for name, parameter in model.named_parameters():
            if name in self.initial:
                value = parameter.detach().float().cpu()
                assert torch.isfinite(value).all(), f"Non-finite adapter tensor: {name}"
                changes[name] = float((value - self.initial[name].float()).abs().max())
        self.report.update(completed_steps=state.global_step,
                           changed_tensors=sum(v > 0 for v in changes.values()),
                           max_parameter_change=max(changes.values()), all_adapter_values_finite=True)
        self.save(args)
        assert self.report["changed_tensors"] > 0, "Optimizer did not update any adapter tensor"
        print("AUDIT: " + json.dumps({k: v for k, v in self.report.items() if k != "steps"}), flush=True)


if __name__ == "__main__":
    run_exp(config, callbacks=[AuditCallback()])
