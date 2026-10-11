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

### 8.1 选择复现目标

|目标|入口|能证明什么|
|---|---|---|
|检查已保存的记录|第 2、7 节的数据检查和指标重算|发布数据、指纹和保存评分一致；不是重新推理或训练|
|复评固定 adapter|第 3 节及下方命令|已发布权重在当前环境的生成表现|
|从基础模型重新训练 v2|本节独立复现入口|重新执行普通/困难组训练、验证选择和最终评估|

复评两组已发布 adapter 时，完成第 1 节环境与模型准备后运行：

```bash
.venv/bin/python scripts/verify_nameplates_v2.py
.venv/bin/python scripts/evaluate_nameplates_v2.py \
  --dataset-dir data/nameplates_v2 --split test \
  --adapter artifacts/nameplate_v2_01/control \
  --output-dir outputs/repro_published_control_test_01
.venv/bin/python scripts/evaluate_nameplates_v2.py \
  --dataset-dir data/nameplates_v2 --split test \
  --adapter artifacts/nameplate_v2_01/hard \
  --output-dir outputs/repro_published_hard_test_01
```

### 8.2 独立重训入口

从全新克隆开始执行第 1 节安装与下载。直接使用仓库图片，不需要重新绘图或准备 macOS 字体。以下入口不依赖历史训练目录、v1 第 50/100 步 checkpoint 或合并模型：

```bash
# 预览：只列出命令顺序；不安装依赖、不创建目录，也不代表环境检查通过
python3 scripts/reproduce_nameplates_v2.py --dry-run

# 检查：验证文件存在、依赖版本、可用 MPS、框架提交和固定模型文件哈希
.venv/bin/python scripts/reproduce_nameplates_v2.py --check-only

# smoke：数据和 token 检查，一次优化器更新，两条验证集生成
.venv/bin/python -u scripts/reproduce_nameplates_v2.py \
  --smoke --run-dir outputs/v2_repro_smoke_01

# 完整运行：两组独立训练，每组 150 次更新
.venv/bin/python -u scripts/reproduce_nameplates_v2.py \
  --run-dir outputs/v2_repro_full_01
```

`--run-dir` 相对仓库根目录解析，必须是 `outputs/` 下尚不存在的子目录。已有目录会被拒绝，不自动续训或覆盖。可以同时指定 `--dry-run --smoke` 预览短流程。`--check-only` 不开始训练或推理；文件内容及图片解码检查由正式执行时的验证阶段完成。

预检查失败会列出缺少的环境、文件或版本。缺少环境执行 `bash scripts/setup.sh`；缺少模型执行 `.venv/bin/python scripts/download_model.py`。入口只支持已记录的 MPS 配置，不会静默回退到 CPU。

完整队列依次执行：v1/v2 数据检查、两组原生 token 检查、困难组单步试训、普通组训练与三个 checkpoint 的生成验证、困难组训练与三个 checkpoint 的生成验证、固定选择、四个模型的新测试集评估、两组的旧验证集回归检查，以及指标复算和父组 bootstrap。

两组均从相同基础模型新建 rank 4 LoRA，seed=42，batch=1，梯度累积 8，最多 150 次优化器更新。800 条记录对应约 1.5 轮；每 50 步保存，分别在 200 条验证样本上生成。选择顺序固定为：严格全对数最多、缺失字段补值数最少、有值字段置空数最少、步数最少。两组权重和配置指纹写入 `selection.json` 后，才开始最终测试。

测试模型为基础模型、仓库发布的 v1 adapter、本次普通组和困难组所选 adapter。最终测试集为 200 张、50 个父组；统计口径与原实验相同。重新训练得到的分数或浮点结果不要求逐字节等于原实验，应报告实际观察到的结果。

### 8.3 输出和完成标准

|产物|内容|
|---|---|
|`plan.json` / `preflight.json`|执行步骤、依赖版本和模型哈希|
|`configs/`|本次运行配置；输出目录指向本次新训练目录|
|`protocol.json`|开始前固定的数据、代码、配置指纹和选择规则|
|`status.json` / `logs/`|模式、阶段、子进程、完成与失败状态；每 30 秒更新运行状态|
|`train_control/` / `train_hard/`|新训练的 checkpoint、训练审计和曲线|
|`validation_candidates.json` / `selection.json`|候选指标、选择时间及所选权重指纹|
|各评估目录|原始预测、运行配置、严格评分与规则后处理指标|
|`REPORT.md` / `final_summary.json`|完整模式的结果总表、分组统计与配对 bootstrap|
|`smoke_summary.json`|仅短流程：1 次更新、2 条验证生成；明确不是完整实验|

完整复现的完成条件是 `status.json` 中 `mode=full` 且 `status=completed`，并生成最终报告。smoke 的 `completed` 只表示短流程完成，不能当成准确率测量或两组完整重训通过。

运行中保持供电和唤醒。查看 `status.json` 和当前阶段日志，不要重复启动；新旧 v2 队列共用文件锁。失败会保留现场和失败阶段；处理后使用新目录重跑。普通与困难模型、不同验证阶段都在独立子进程中串行运行。

### 8.4 可选：v1 历史诊断与合并验证

历史脚本 `run_nameplate_v2_experiment.py` 及已发布结果保留原样，以保持原实验指纹。该脚本依赖原机历史目录；新克隆应使用上面的 `reproduce_nameplates_v2.py`。

v1 历史 checkpoint 的预测和对比已保存在 `results/v2_checkpoint50_val_01`、`results/v2_checkpoint100_val_01`、`results/nameplate_lora_val_01` 与 `results/v1_checkpoint_comparison_v2.json`。原第 50/100 步权重不随仓库发布；如需重新训练后比较，先完成第 4 节，再显式评估本次三个 checkpoint：

```bash
for step in 50 100 150; do
  .venv/bin/python scripts/evaluate_nameplates.py \
    --split val --adapter "outputs/repro_lora_01/checkpoint-$step" \
    --output-dir "outputs/repro_v1_step${step}_val_01"
done
```

以上比较属于新训练轨迹，不复用原第 150 步预测来代表新 checkpoint。

若要检查已发布 v1 adapter 的完整合并输出，先完成第 5.1 节导出，再运行：

```bash
.venv/bin/python scripts/evaluate_nameplates_v2.py \
  --dataset-dir data/nameplates_v1 --split test \
  --model outputs/published_nameplate_merged_01 \
  --reference-dir results/nameplate_lora_test_01 --require-match \
  --output-dir outputs/repro_published_merged_test_01
```

该命令检查全部 100 条旧测试记录；它与 v2 两组重训相互独立。出现数值或生成差异时保留报告并调查，不修改历史参考答案或放宽原评分规则。

### 8.5 入口验证范围（2026-10-11）

新增入口已在独立克隆目录中运行 `--smoke`：开始时不存在 `outputs/` 和 `logs/`，仅复用已安装的固定版本环境、框架和模型缓存，没有复用历史训练或评估产物。v1/v2 数据检查、两组 token 检查、1 次优化器更新及 2 条验证集生成均完成；训练审计记录 544,768 个可训练参数、56 个张量发生变化，adapter 数值全部有限。

仓库 28 项单元测试及原历史实验发布校验也通过。测试覆盖无环境预览、输出目录保护、失败停止、验证选择先于测试、所选权重变化检测，以及用已保存预测复算完整报告。此次未重新安装依赖，也未执行两组各 150 步的完整重训；单步试训的两条输出不用于判断模型效果。

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

## 10. 原历史队列的校验与本地归档

本节只适用于原 `run_nameplate_v2_experiment.py` 的完整目录，依赖其中的 v1 合并检查等附加产物。新独立入口在结束前自行复算评分、检查选择顺序及模型指纹，并输出第 8.3 节所列报告；不要将新目录传给历史归档工具。

原历史队列状态为 `completed` 后执行：

```bash
.venv/bin/python scripts/publish_nameplate_v2.py --check-only
.venv/bin/python scripts/publish_nameplate_v2.py
```

归档工具会复算所有候选 checkpoint、最终测试和旧验证集回归的评分，核对预先固定的选择规则、权重指纹和测试启动时间，重算父组 bootstrap，并检查训练审计。未完成的实验、已有目标目录或指标不一致都会导致退出。

默认将报告、逐条预测、训练曲线和来源指纹写入 `results/v2_experiment_01`，两个所选 adapter 写入 `artifacts/nameplate_v2_01/{control,hard}`。原始运行目录保持不变，只在 JSON/JSONL 发布副本中移除仓库绝对路径前缀。`publication_manifest.json` 分别记录源文件和发布副本的哈希。该步骤只做本地归档，不上传或推送 GitHub。
