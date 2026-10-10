# 复现说明

以下命令在仓库根目录运行。已验证平台为 Apple M3 / 16 GiB、macOS 14.3、ARM64、MPS。固定配置未在 CUDA、其他 Mac 或 Linux 上验证。

`results/` 是已完成实验的发布副本，`outputs/` 和 `logs/` 是本地运行目录。新实验使用新的输出目录和报告名称。评估和 token 检查脚本拒绝覆盖已有结果；训练器复用输出目录可能自动恢复 checkpoint，应在运行前确认目录是否存在。

## 1. 安装环境和下载模型

安装 Git 和 uv 后运行：

```bash
bash scripts/setup.sh
.venv/bin/python scripts/download_model.py
```

setup 脚本创建 Python 3.11.17 环境，安装固定依赖，检查 MPS 基础 BF16 参数更新，并确认框架版本。它会创建 `logs/`、`outputs/`；不会开始图文模型训练。

固定来源：

|对象|版本|
|---|---|
|LLaMA-Factory|v0.9.5，`7af909522a951e3ad9f022ea6f88b6755257eaa5`|
|Qwen2-VL-2B-Instruct|`895c3a49bc3fa70a340399125c650a463535e71c`|

下载脚本将基础模型保存到 `models/Qwen2-VL-2B-Instruct`，包括许可证、分词器和图像处理配置。后续训练与评估默认离线加载。

## 2. 检查已有数据和实验报告

```bash
.venv/bin/python scripts/verify_repository.py
```

该命令不加载模型。它检查数据哈希、600 张图片的解码与哈希、答案格式、跨集合图片/型号/模板重复，并从保存的 400 条原始预测重新计算指标。它还检查 adapter、关键脚本的版本指纹与发布文件哈希。

已有数据已注册为 `nameplate_train`、`nameplate_val`、`nameplate_test`，不需要重新生成。数据检查不等于逐图人工校对标签，也不证明真实照片泛化能力。

若要单独研究生成器，可运行：

```bash
.venv/bin/python scripts/generate_nameplates.py \
  --output-dir data/nameplates_regenerated_01
```

这一步需要生成器指定的四种 macOS 系统字体，不随仓库提供。字体版本和 Pillow 版本会影响渲染字节，不能仅靠随机种子保证其他环境图片完全一致。以上命令不传 `--register`，不会改写当前实验的数据注册。原目录存在时，生成器拒绝覆盖。

## 3. 复核已发布 adapter

完成基础模型下载后，可直接运行：

```bash
.venv/bin/python -u scripts/evaluate_nameplates.py \
  --split test \
  --adapter artifacts/nameplate_lora_01 \
  --output-dir outputs/published_adapter_test_01
```

它不训练模型，每条记录保存原始回答、标准答案、评分、终止原因与耗时。该路径用于复核已发布的固定模型；开展新实验时，应先用训练集和验证集开发，再在模型固定后使用测试集。

## 4. 从基础模型重新训练

### 4.1 验证集基线

```bash
.venv/bin/python -u scripts/evaluate_nameplates.py \
  --split val \
  --output-dir outputs/repro_base_val_01
```

### 4.2 原生预处理检查

```bash
.venv/bin/python scripts/check_tokens.py \
  --config configs/train_nameplates_mps.yaml \
  --output logs/repro_token_validation_01.json
```

本轮参考结果为训练 400、验证 100 条，序列长度 512–521 tokens，无空监督标签，无 labels/input_ids 长度不一致。

### 4.3 单步试训

```bash
.venv/bin/python -u scripts/train.py \
  --config configs/train_nameplates_mps.yaml \
  --smoke \
  --output-dir outputs/repro_smoke_01
```

`--smoke` 完成 1 次优化器更新，保留梯度累积 8，关闭验证并保存参数。检查输出中的 AUDIT：completed_steps 为 1、changed_tensors 大于 0、all_adapter_values_finite 为 true。

### 4.4 完整训练

```bash
set -o pipefail
.venv/bin/python -u scripts/train.py \
  --config configs/train_nameplates_mps.yaml \
  --output-dir outputs/repro_lora_01 \
  2>&1 | tee logs/repro_train_01.log
```

完整训练从基础模型新建 LoRA，不接着单步产物训练。400 条数据、batch size 1、梯度累积 8，对应每轮 50 次更新，3 轮共 150 次；第 50、100、150 步验证并保存。

参考耗时约 2 小时 19 分，包含训练器执行期间的验证等开销。不同环境的耗时和浮点结果可能不同。

### 4.5 验证集生成评估

```bash
.venv/bin/python -u scripts/evaluate_nameplates.py \
  --split val \
  --adapter outputs/repro_lora_01 \
  --output-dir outputs/repro_lora_val_01
```

比较同一集合、同一生成协议的基线和 LoRA 结果。验证 loss 与生成准确率分别统计。

### 4.6 固定模型后进行测试

```bash
.venv/bin/python -u scripts/evaluate_nameplates.py \
  --split test \
  --output-dir outputs/repro_base_test_01
.venv/bin/python -u scripts/evaluate_nameplates.py \
  --split test \
  --adapter outputs/repro_lora_01 \
  --output-dir outputs/repro_lora_test_01
```

测试错误用于报告当前固定版本的表现。如果据此调整训练，原测试集就已参与开发，应另外准备最终评估数据。

## 5. 合并和重载

### 5.1 导出仓库中的 adapter

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/llamafactory-cli export configs/export_published_nameplates.yaml
.venv/bin/python -u scripts/verify_nameplate_export.py \
  --model outputs/published_nameplate_merged_01 \
  --reference-dir results/nameplate_lora_test_01 \
  --output logs/published_reload_check_01.json
```

该配置读取 `artifacts/nameplate_lora_01`。导出前确认 `outputs/published_nameplate_merged_01` 不存在，并预留完整 BF16 权重的空间。重载脚本从参考预测中选取四类样例；本轮为序号 1、3、12、19，包含两个已知错误。

`export_device: auto` 在本机使用 MPS。早期 demo 曾观察到 BF16 CPU/MPS 合并的数值差异，不应假定其他设备导出的回答逐字相同。

### 5.2 导出重新训练的 adapter

复制 `configs/export_published_nameplates.yaml` 为一个新配置，将 `adapter_name_or_path` 改为 `outputs/repro_lora_01`，将 `export_dir` 改为新的目录。执行导出后，重载检查的 `--reference-dir` 使用 `outputs/repro_lora_test_01`，与本次训练的预测对应。

保留完整模型的 tokenizer、processor、聊天模板和基础模型 LICENSE。重载时继续使用 qwen2_vl、BF16、301056 像素上限、贪心解码和 128 个新 token 上限。

## 6. 早期官方 demo

铭牌实验不需要下载官方 demo。若要复查早期配置，可运行：

```bash
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/train.py --smoke --output-dir outputs/demo_smoke_02
```

`prepare_data.py` 从固定官方提交获取图片与对话，生成 4 条训练记录、2 条留出记录，并合并数据注册表。官方图片未随本仓库再次分发。默认 `train_mps.yaml` 仅用于该 demo，不能代替铭牌配置。
