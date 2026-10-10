# Multimodal LoRA Lab

在 Apple M3 / 16 GiB 上，使用 LLaMA-Factory 对 Qwen2-VL-2B-Instruct 进行多模态 LoRA SFT。任务是从设备铭牌图片中提取型号、电压和功率，直接输出 JSON。

本仓库包含实验数据、训练配置、评估脚本、最终 LoRA adapter 和过程记录。实验完成于 2026-10-10。

## 任务示例

![合成设备铭牌](data/nameplates_v1/images/99f58bad21059c67.png)

标准答案：

```json
{"型号":"SV-230","电压":"110V","功率":null}
```

型号按图抄录；电压去除 AC/DC 和空格；功率去除空格。字段未出现、空白或标为 `--`、`N/A` 时输出真正的 JSON `null`。

## 实验结果

600 张合成图片按训练 400、验证 100、测试 100 划分。型号组、模板 ID 和字节相同的图片不跨集合重复。LoRA 训练 3 轮、150 次优化器更新，训练器报告耗时约 2 小时 19 分。

|独立测试集（100 张）|基础模型|LoRA|
|---|---:|---:|
|严格格式合规率|0%|100%|
|严格全对率|0%|97%|
|仅去掉外围代码围栏后的全对率|20%|97%|
|型号正确率|90%|100%|
|电压正确率|27%|97%|
|功率正确率|41%|99%|
|缺失字段正确数|0/30|28/30|

严格全对要求原始回答符合 JSON 与字段规则，并且三个值全部正确。诊断计分只去除整段外围的代码围栏，不修正单位、字段值或 null 类型。

验证集严格全对率为 99/100。完整模型合并导出后，4 条固定样例的原始回答与终止原因均与 adapter 推理一致；没有对完整导出模型另做全量 100 条测试。

## 使用已训练的 adapter

已验证环境为 macOS 14.3 / ARM64 / MPS。以下命令在仓库根目录执行，需要 Git、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和可用的 MPS 设备。

```bash
git clone https://github.com/jiangzyOvO/multimodal-lora-lab.git
cd multimodal-lora-lab
bash scripts/setup.sh
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/verify_repository.py
.venv/bin/python scripts/evaluate_nameplates.py \
  --split test \
  --adapter artifacts/nameplate_lora_01 \
  --output-dir outputs/published_adapter_test_01
```

基础模型约 4.42 GB，由下载脚本获取固定版本。仓库内 adapter 权重约 2.2 MB，需要配合基础模型及其 tokenizer/processor 使用。完整合并权重、训练 checkpoint 和 Python 环境不在 Git 仓库中。

从数据检查、基线到训练、评估、合并的逐步命令见 [复现说明](docs/REPRODUCING.md)。脚本的默认配置中也保留了早期官方 demo；铭牌训练须显式传入 `--config configs/train_nameplates_mps.yaml`。

## 实测配置

|项目|设置|
|---|---|
|模型|Qwen2-VL-2B-Instruct，固定 revision|
|框架|LLaMA-Factory v0.9.5，固定提交|
|环境|Python 3.11.17，PyTorch 2.8.0，Transformers 4.57.6|
|LoRA|rank 4，alpha 8，q_proj/v_proj，544,768 个可训练参数|
|冻结模块|视觉编码器和多模态投影模块|
|训练|BF16，batch size 1，梯度累积 8，3 轮|
|学习率|1e-4，10 步预热，余弦调度|
|输入|672×448 图片，cutoff_len 1024|
|生成|关闭采样，max_new_tokens 128|

## 文件索引

|位置|内容|
|---|---|
|[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)|阶段进度、兼容性问题、关键发现和错误分析|
|[docs/REPRODUCING.md](docs/REPRODUCING.md)|安装、评估、训练、导出的复现命令|
|[data/nameplates_v1](data/nameplates_v1/README.md)|600 张图片、标签、划分和绘图规格|
|[configs](configs)|训练与导出配置，包括保留的失败配置|
|[scripts](scripts)|环境检查、数据生成、训练、评估和检查脚本|
|[artifacts/nameplate_lora_01](artifacts/nameplate_lora_01/README.md)|最终 adapter、版本指纹和基础模型许可证|
|[results](results/README.md)|原始预测、指标、训练曲线及关键日志的发布副本|
|[requirements.lock.txt](requirements.lock.txt)|实测环境的完整依赖版本|

## 已知问题和范围

测试集有三条失败样本：两条把空白电压补成 220V；另一条将功率 1800W 填入电压字段，并漏掉功率。图片、答案和分类见 [实验记录](EXPERIMENT_LOG.md#81-测试失败案例)。

所有图片均为合成图，集合之间仍共享字体和绘图组件。97% 不代表真实拍摄铭牌的准确率。数据生成脚本依赖 macOS 系统字体；使用仓库已有图片不需要重新生成。其他硬件、量化部署、Ollama 和 verl/RL 尚未验证。

## 来源

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：训练框架，按固定版本安装。
- [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)：基础模型。
- 第三方来源、版本和许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
