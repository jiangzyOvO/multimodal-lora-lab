# 实验结果与发布副本

本目录保存 2026-10-09 至 2026-10-10 已完成实验的结果，不是发布时重新训练得到的数据。

|目录|内容|
|---|---|
|[nameplate_base_val_01](nameplate_base_val_01)|基础模型验证集：100 条|
|[nameplate_lora_val_01](nameplate_lora_val_01)|LoRA 验证集：100 条|
|[nameplate_base_test_01](nameplate_base_test_01)|基础模型测试集：100 条|
|[nameplate_lora_test_01](nameplate_lora_test_01)|LoRA 测试集：100 条|
|[training](training)|训练与验证指标、trainer 状态、参数更新审计、曲线|
|[checks](checks)|环境、模型来源、单步审计、token 检查和导出重载检查|
|[logs](logs)|关键训练、导出和失败日志|
|[demo](demo)|早期官方 demo 的训练、合并精度与重载证据|
|[analysis_v2](analysis_v2)|基于 v1 原始预测的追加诊断：规则后处理基线、缺失值指标、分组与逐样本对照；未重新推理|
|[v1_checkpoint_comparison_v2.json](v1_checkpoint_comparison_v2.json)|第 50/100/150 步的同验证集对比及新增预测发布指纹|
|[v2_checkpoint50_val_01](v2_checkpoint50_val_01)、[v2_checkpoint100_val_01](v2_checkpoint100_val_01)|原 v1 训练第 50、100 步新增生成评估；目录中的 v2 指本轮审计，不是新数据训练|
|[v2_data_check_01.json](v2_data_check_01.json)|1,200 张 v2 新图片与两组训练数据检查|
|[v2_visual_spotcheck_01.json](v2_visual_spotcheck_01.json)|四张图片的助手视觉抽查与独立转录记录；不是全量人工审核|
|[v2_merged_v1_test_01](v2_merged_v1_test_01)|完整合并模型的 100 条新增推理及与 adapter 的逐条对照，全部一致|
|[v2_experiment_01](v2_experiment_01/REPORT.md)|已完成的普通/困难扩充对照：选择规则、六次 checkpoint 验证、四模型新测试、旧验证集回归、训练记录和独立归档指纹|

每个生成评估目录包含 `run.json`（输入版本、配置与实际设备）、`predictions.jsonl`（逐条原始预测及评分）和 `summary.json`（指标）。诊断计分只移除外围代码围栏，不修复字段值。

## 发布时的处理

原始本地报告不变。发布副本中的项目绝对路径转换为相对路径，其他个人主目录前缀替换为 `<HOME>/`。没有改动模型回答、标准答案、指标、随机种子、时间戳或实验版本指纹。

[publication_manifest.json](publication_manifest.json) 记录每份副本对应的原始相对路径、原始 SHA-256、发布 SHA-256，以及是否发生路径替换。路径替换会改变文件字节，因此副本自身的哈希不必等于原文件哈希。报告中已有的数据、脚本和参考预测哈希仍指向原实验对象。

从仓库根目录运行：

```bash
.venv/bin/python scripts/verify_repository.py
```

检查过程会从原始回答重算四组指标，并核对发布文件、原始数据、adapter 和关键评估脚本的哈希，不调用模型。该检查确认保存结果内部一致，不替代重新推理或人工图片审核。

## 指标范围

验证集与测试集分别计分。最初的 97% 来自固定 adapter 的 100 条测试预测；首轮导出只检查四条样例。随后追加的 `v2_merged_v1_test_01` 完成全量 100 条推理，原始输出、终止原因和评分全部一致，合并模型的严格全对率同为 97%。

`train_runtime` 是训练器汇总耗时，包含训练中验证等开销。参数审计中的 MPS 内存是在优化步骤结束时采样，不能当作全程精确峰值。

v2 最终测试使用另行生成的 200 张图片、50 个父组：原 v1 adapter、普通扩充和困难扩充的严格全对分别为 191/200、193/200 和 198/200。两组 v2 训练均使用 800 条记录、150 次更新，只依据验证集选择 checkpoint；不能把 v1 的 97/100 与 v2 的 198/200 当作同一测试集上的前后对比。v2 归档的来源及发布副本哈希由 `v2_experiment_01/publication_manifest.json` 独立记录。

`analysis_v2` 不属于原 `publication_manifest.json` 中的历史发布副本；每份派生报告单独记录数据、预测和分析代码的 SHA-256。分析规则及其事后性质写在报告 `policy` 中，严格指标保持原定义。null precision 的分母是预测为 null 的字段数，recall 的分母是实际缺失字段数；分母为零时报告 JSON null。`spurious_value_rate` 统计实际缺失字段被填入非 null 值的比例，`false_null_rate` 统计实际有值字段被置为 null 的比例。不可解析和漏键单列，并保留在准确率及 null recall 的分母中。
