# Qwen2-VL-2B 铭牌提取 LoRA

本目录保存 2026-10-09 完成的最终 LoRA adapter。它基于 Qwen2-VL-2B-Instruct，任务为读取图片并输出型号、电压、功率三个 JSON 字段。

|项目|值|
|---|---|
|基础模型|`Qwen/Qwen2-VL-2B-Instruct`|
|基础模型 revision|`895c3a49bc3fa70a340399125c650a463535e71c`|
|训练框架|LLaMA-Factory v0.9.5|
|LoRA|rank 4、alpha 8、q_proj/v_proj|
|训练|400 条合成图片，3 轮，150 步|
|可训练参数|544,768|
|权重文件大小|2,195,400 字节|

`adapter_model.safetensors` 和 `adapter_config.json` 均为原始训练产物的逐字节副本，未改写配置。配置中的基础模型路径为 `models/Qwen2-VL-2B-Instruct`，与项目下载脚本对应；基础模型的固定 revision 由下载脚本控制。分词器和图像处理器从该基础模型获取。

在仓库根目录安装环境、下载基础模型后运行：

```bash
.venv/bin/python scripts/evaluate_nameplates.py \
  --split test \
  --adapter artifacts/nameplate_lora_01 \
  --output-dir outputs/published_adapter_test_01
```

100 张合成测试图严格全对率为 97%，缺失字段正确 28/30。已知错误为两次空白电压补值和一次字段错配；不能据此推断真实照片准确率。详细结果见 [测试摘要](../../results/nameplate_lora_test_01/summary.json)。

## 版本指纹

```text
adapter_model.safetensors
2d5397a9cacf0b5ad50ab05c94125832a0712a9db1ded7b9b51fe3bece52ba53

adapter_config.json
03a782b15bdda808340b1888574a41655910c8108aa08b9f187fb2c911a356c1
```

本目录只包含 adapter，不含完整模型。合并配置为 [export_published_nameplates.yaml](../../configs/export_published_nameplates.yaml)。本机使用 BF16 / MPS 合并；不同设备的浮点差异可能改变生成结果。

附带的 [LICENSE](LICENSE) 为基础模型原始 Apache-2.0 许可证。adapter 是本项目训练产生的衍生产物，未替换或修改基础模型原文件。第三方来源见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
