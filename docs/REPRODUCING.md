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

## 7. 追加诊断与规则后处理基线

此阶段只读取保存的预测，不加载模型。原始严格评分脚本保持不变，新增规则后处理和缺失值双向错误统计。默认读取 `results/nameplate_{base,lora}_{val,test}_01`，并核对两次评估的输入、样本顺序、提示、标准答案、种子、精度和推理配置。

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/analyze_nameplate_predictions.py \
  --split val --output outputs/analysis_replay_01/val.json
.venv/bin/python scripts/analyze_nameplate_predictions.py \
  --split test --output outputs/analysis_replay_01/test.json
.venv/bin/python scripts/verify_repository.py
```

输出路径已经存在时会报错，请换用新目录。`--base-dir`、`--lora-dir` 可指定其他完整评估目录；`--dataset-dir` 指向包含对应集合 JSON 和 `metadata.jsonl` 的数据目录，标签必须符合当前整数 V/W 规范。

报告包含：原始严格指标、仅去围栏指标、规则处理后的指标、null precision/recall、错误补值率、错误置空率、各切片样本数和逐样本改善/退化。没有预测 null 时，null precision 为 null，而不是 0 或 1。没有缺失字段时，null recall 和错误补值率为 null。

报告使用固定策略 `nameplate_normalization_v1`，不读取标准答案来修正预测，也不修改源预测。v1 测试集已经参与错误分析，后续依据这些发现修改数据或模型时，需准备新的最终测试集。生成的分析报告保留输入文件和分析脚本指纹；修改规则应使用新的策略版本和输出目录。

## 8. v2 数据与串行实验

[v2 数据说明](../data/nameplates_v2/README.md)列出普通扩充、困难扩充和新评估集合。先运行：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/verify_nameplates_v2.py
```

v1 第 50、100 步 checkpoint 使用原评估脚本运行验证集，分别输出到 `outputs/v2_checkpoint50_val_01`、`outputs/v2_checkpoint100_val_01`。第 150 步权重与发布 adapter 哈希一致，复用已保存的验证结果。随后执行：

```bash
.venv/bin/python -u scripts/run_nameplate_v2_experiment.py \
  --run-dir outputs/v2_experiment_01
```

队列依次执行：v1/v2 数据检查、合并模型完整 100 条对照、两组原生 token 检查、困难组单步试训、普通组训练与三个 checkpoint 的生成验证、困难组训练与三个 checkpoint 的生成验证、固定选择、四个模型的新测试集评估、两个新模型的旧验证集回归检查。MPS 模型任务逐个启动，前一个进程退出后才运行下一个。

两组配置均从基础模型新建 rank 4 LoRA，seed=42，batch=1，梯度累积 8，最多 150 次优化器更新。800 条训练记录下这对应约 1.5 轮。训练期间每 50 步保存，不额外计算 teacher-forced 验证 loss；各 checkpoint 单独运行 200 条生成验证用于选择。

选择顺序预先固定为：严格全对数最多、缺失字段补值数最少、有值字段置空数最少、步数最少。数据和代码指纹保存在 `protocol.json`，两个所选 adapter 的指纹保存在 `selection.json`；两者均在最终测试推理之前产生。最终测试包含基础模型、旧 adapter、普通扩充模型、困难扩充模型。单个训练种子的结果不包含训练随机性的不确定性。

`status.json` 每 30 秒更新当前阶段、子进程和已完成阶段；各阶段原始日志在 `logs/`，逐图预测保存在相应评估目录。出现错误时队列停止并记录失败阶段，不覆盖目录，也不自动恢复部分训练。完整结束后生成 `final_summary.json`，包含分组统计和以 50 个父组为重采样单位的配对 bootstrap 区间。

长任务期间保持机器供电和唤醒。请先查看 `status.json`，不要重复启动同一个实验。脚本通过文件锁拒绝同时启动多个 v2 队列。

## 9. 真实铭牌采集与标注

目前没有真实照片，真实照片测试尚未开展。首批目标为 100–200 张，记录设备/型号系列标识、来源、拍摄条件和原始图片。同一设备的不同角度、裁剪、压缩或重拍图属于同一组，不能跨训练、验证和最终测试集合。

采集覆盖清晰正面、倾斜、反光、小字、磨损及多组额定参数。保存原始照片，另行记录预处理步骤。至少由一人标注，并由另一人或独立的第二轮核对原图；保留字段原文和标准化结果，不能用待评估模型的回答直接当标签。

逐字段区分以下状态：

|状态|处理|
|字段可见且值清晰|按当前规范转录，记录原文|
|字段确实缺失、空白或明确缺失标记|目标值为 JSON null|
|反光、遮挡、模糊导致无法辨认|标为 unreadable，单列待复核；不伪装成字段缺失|
|同图多组电压/功率且目标不明确|标为 ambiguous，先确定任务定义|
|小数、范围或 kW 等超出现有规范|标为 out_of_scope；扩展 schema 和评分器后再评估|

首轮先冻结当前模型与提示，对独立真实测试集做迁移评估；如随后据此开发新模型，应另外保留新的最终测试组。报告同时列出纳入数量及 unreadable、ambiguous、out_of_scope 数量，避免仅展示筛选后准确率。

## 10. 完成后的校验与本地归档

实验队列状态为 `completed` 后执行：

```bash
.venv/bin/python scripts/publish_nameplate_v2.py --check-only
.venv/bin/python scripts/publish_nameplate_v2.py
```

归档工具会复算所有候选 checkpoint、最终测试和旧验证集回归的评分，核对预先固定的选择规则、权重指纹和测试启动时间，重算父组 bootstrap，并检查训练审计。未完成的实验、已有目标目录或指标不一致都会导致退出。

默认将报告、逐条预测、训练曲线和来源指纹写入 `results/v2_experiment_01`，两个所选 adapter 写入 `artifacts/nameplate_v2_01/{control,hard}`。原始运行目录保持不变，只在 JSON/JSONL 发布副本中移除仓库绝对路径前缀。`publication_manifest.json` 分别记录源文件和发布副本的哈希。该步骤只做本地归档，不上传或推送 GitHub。
