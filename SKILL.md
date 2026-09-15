---
name: transcribe-video-long
description: 对本地长视频进行可恢复的多语言 Whisper 分段转录，并可用本机 GGUF 翻译模型输出精炼的逐句双语稿和双语 SRT。
---

# 长视频多语言双语转录

## 入口

运行时脚本：

```bash
python3 scripts/transcribe_video_long.py VIDEO \
  --language auto \
  --translate \
  --translation-model "$HOME/.cache/hy-mt/HY-MT1.5-1.8B-Q4_K_M.gguf"
```

`--language` 可填写 `en`、`ja`、`ko` 等 Whisper 语言代码；不确定时使用 `auto`。中文视频不要加 `--translate`，或使用 `--language zh`。

Whisper 模型默认读取 ~/.local/share/whisper.cpp/ggml-large-v3-turbo-q5_0.bin；也可以用 --whisper-model 指定本地模型路径。

如果用户给的是视频 URL，不把下载逻辑塞进本脚本；先按 `video-downloader` skill 把 URL 下载到本地，再把下载后的本地视频路径传给本脚本。对用户表现为一条工作流，但职责边界保持为“下载 skill 负责取文件，本 skill 负责本地视频转写翻译”。

已有输出可单独复检：

```bash
python3 scripts/transcribe_video_long.py validate OUTPUT_DIR \
  --stem VIDEO_BASENAME \
  --expect-bilingual
```

## 产物位置

不显式传 -o/--output-dir 时，默认把本次转录放在当前仓库的集中目录：

~~~text
.teaching-video-runtime/outputs/transcribe-video-long/<视频名>-<源文件 sha256 前 10 位>/
~~~

这样不会散落到视频下载目录，也能避免同名不同视频互相覆盖。若用户指定 -o/--output-dir，则以用户指定目录为准。

默认可见产物只有：

~~~text
<视频名>.sentences.md
<视频名>.srt
manifest.json
~~~

可恢复缓存保留在该输出目录下的隐藏子目录 .segments/；成功后默认清理音频缓存。

## 流程

1. 本地转写脚本检查 `ffmpeg`、`whisper-cli`、`llama-completion` 和 Whisper/Hunyuan 模型；如果输入是 URL，下载阶段由 `video-downloader` 先完成。
2. 输入视频或下载后的视频保留不变，所有结果写入独立输出目录。
3. 提取 16 kHz 单声道音频，按默认 15 分钟切片。
4. 已完成的 Whisper 片段跳过，支持中断后续跑。
5. 合并原文字幕并修正分段时间轴；默认生成双语 SRT，TXT/VTT 仅按需生成。
6. `--translate` 时默认按相邻字幕窗口调用 Hunyuan-MT，利用上下文翻译但仍使用原 cue 时间轴，把原文和中文写入同一字幕块；需要完全逐 cue 独立翻译时使用 `--translation-mode cue`。
7. 写入 `manifest.json`，记录源视频、Whisper 模型、翻译模型、语言、切片参数、工具版本和输出文件。
8. 翻译缓存写入输出目录 `.segments/translations.json`，按翻译模型、语言、prompt 和推理参数隔离；每条翻译原子落盘。
9. 默认只交付 `.sentences.md` 和 `.srt`：逐句稿用于阅读，SRT 用于播放器字幕。`.txt`、`.vtt`、`.sentences.txt`、`.sentences.json`、独立 `validation.json` 仅在显式 `--emit-*` 或 `--full` 时产出。
10. 逐句稿先把 Whisper cue 串成全文，按句末标点切成自然句，再用 cue 内字符位置回推近似时间范围。
11. 校验结果默认内嵌到 `manifest.json`；仅调试或外部流水线需要独立文件时使用 `--emit-validation-json`。

## 完成标准

只有以下检查全部通过，才能报告双语转录完成：

- Whisper 片段数量与合并结果一致；
- 默认交付产物 `.sentences.md`、`.srt` 和 `manifest.json` 均存在；
- SRT 时间轴单调且覆盖各分段偏移；
- 双语输出每个 cue 同时包含原文和中文；
- 逐句稿存在且按自然句/短段保留原文、中文和时间范围；
- 翻译缓存可被同模型、同语言、同 prompt 的再次运行复用，且不会跨模型/语言误复用；
- `manifest.json` 记录源视频 hash、模型 hash 与关键参数；
- `manifest.json` 内的 validation 显示 `ok: true`；
- 至少用一条实际译文验证 `llama-completion` 返回非空中文；
- 长视频中断后重跑不会重复识别已完成片段。

当前实现包含“逐字幕 cue 对齐字幕”和“自然句双语稿”，不是词级时间戳对齐。

## 网络、代理与沙箱规则

- 不把 Codex 沙箱里的代理错误当成宿主机网络错误。
- 普通执行环境可能无法访问宿主机 Clash 的 `127.0.0.1`；需要下载时，先检查错误，再使用受控网络执行或让用户在宿主机执行。
- 模型下载必须先查询官方文件大小和 SHA256，下载到临时文件，禁止多个续传进程同时写最终文件。
- 只在 `GGUF` magic、文件大小和 SHA256 都通过后替换最终模型。
- `llama-cli` 的端口错误、沙箱 Metal 错误和模型文件损坏是三类不同问题，分别诊断。
