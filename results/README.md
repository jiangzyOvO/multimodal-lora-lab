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

验证集与测试集分别计分。97% 来自固定 adapter 的 100 条测试预测；完整模型导出后只检查四条样例的原始输出和终止原因一致性。

`train_runtime` 是训练器汇总耗时，包含训练中验证等开销。参数审计中的 MPS 内存是在优化步骤结束时采样，不能当作全程精确峰值。
