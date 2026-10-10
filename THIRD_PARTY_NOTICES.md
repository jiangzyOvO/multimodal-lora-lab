# 第三方来源与许可

## Qwen2-VL-2B-Instruct

- 来源：[Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)。
- 固定 revision：`895c3a49bc3fa70a340399125c650a463535e71c`。
- 基础模型原始许可：[Apache License 2.0](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/895c3a49bc3fa70a340399125c650a463535e71c/LICENSE)。
- 原文副本：[artifacts/nameplate_lora_01/LICENSE](artifacts/nameplate_lora_01/LICENSE)。

仓库中的 LoRA adapter 是在该基础模型上使用合成铭牌数据训练的衍生产物。基础模型权重不随 Git 仓库发布，由下载脚本获取。完整模型导出时需保留基础模型许可证及相关文件。

## LLaMA-Factory

- 来源：[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。
- 固定版本：v0.9.5，提交 `7af909522a951e3ad9f022ea6f88b6755257eaa5`。
- 许可：[Apache License 2.0](https://github.com/hiyouga/LLaMA-Factory/blob/7af909522a951e3ad9f022ea6f88b6755257eaa5/LICENSE)。

框架按原始版本安装，未修改源码，也未将源码副本加入本仓库。早期 demo 图片和对话由 `scripts/prepare_data.py` 从官方固定提交获取，未再次分发图片。

## 字体与其他依赖

合成图片由本项目脚本绘制，使用 macOS 系统字体。字体二进制文件不包含在仓库中；生成记录中的字体路径和哈希用于描述渲染环境，不授予字体使用或再分发许可。

Python 依赖版本列于 `requirements.lock.txt`，各依赖沿用其自身许可。以上第三方许可说明不替代相关原文。
