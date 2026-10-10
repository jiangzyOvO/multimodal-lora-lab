# v2 hard LoRA adapter

Selected on v2 validation at step 150. Synthetic test strict exact: 198/200.

Requires Qwen/Qwen2-VL-2B-Instruct revision 895c3a49bc3fa70a340399125c650a463535e71c and its original tokenizer/processor. rank=4, alpha=8, q_proj/v_proj, BF16 inference.

Single training seed; synthetic data only. See the experiment protocol and raw predictions in results/v2_experiment_01. Base-model license is included.
