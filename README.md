# whisper-video-translator

A local Codex skill and CLI script for long-video transcription and Chinese bilingual subtitle generation.

它把本地长视频转为可恢复的 Whisper 分段转录结果，并可用本地 Hunyuan-MT GGUF 模型生成精炼的逐句双语稿和双语 SRT。

## Features

- Extracts 16 kHz mono audio with ffmpeg.
- Splits long audio into resumable segments.
- Runs whisper-cli per segment and skips completed segments on rerun.
- Generates bilingual SRT with original text and Simplified Chinese in the same cue.
- Generates a sentence-level bilingual Markdown draft for reading.
- Keeps default deliverables lean:
  - VIDEO.sentences.md
  - VIDEO.srt
  - manifest.json
- Stores resumable ASR and translation cache under .segments/.
- Supports optional debug/compatibility outputs via --full or --emit-*.


## Quick start

On macOS:

~~~bash
./scripts/install_macos.sh
python3 scripts/transcribe_video_long.py doctor
~~~

Then run a local video:

~~~bash
python3 scripts/transcribe_video_long.py /path/to/video.mp4   --language en   --translate   --translation-model "$HOME/.cache/hy-mt/HY-MT1.5-1.8B-Q4_K_M.gguf"
~~~

More details:

- docs/INSTALL.md
- docs/DEMO.md

## Requirements

Install these command-line tools first:

- ffmpeg and ffprobe
- whisper-cli from whisper.cpp
- llama-completion from llama.cpp, if using Hunyuan/GGUF translation

You also need local model files:

- a whisper.cpp model, default:
  ~/.local/share/whisper.cpp/ggml-large-v3-turbo-q5_0.bin
- optional Hunyuan-MT GGUF model for translation, for example:
  ~/.cache/hy-mt/HY-MT1.5-1.8B-Q4_K_M.gguf

Model files are intentionally not included in this repository.


## Quick health check

Before processing real videos, run:

~~~bash
python3 scripts/transcribe_video_long.py doctor
~~~

To check a custom model setup:

~~~bash
python3 scripts/transcribe_video_long.py doctor   --whisper-model "$HOME/.local/share/whisper.cpp/ggml-large-v3-turbo-q5_0.bin"   --translation-model "$HOME/.cache/hy-mt/HY-MT1.5-1.8B-Q4_K_M.gguf"
~~~

The doctor command checks Python, ffmpeg, ffprobe, whisper-cli, llama-completion, local model files, and output directory writability.

## Configuration

You can pass model paths as flags, or set environment variables:

~~~bash
export WVT_WHISPER_MODEL="$HOME/.local/share/whisper.cpp/ggml-large-v3-turbo-q5_0.bin"
export WVT_TRANSLATION_MODEL="$HOME/.cache/hy-mt/HY-MT1.5-1.8B-Q4_K_M.gguf"
export WVT_OUTPUT_ROOT=".teaching-video-runtime/outputs/whisper-video-translator"
~~~

Command-line flags override the defaults and are the clearest option for one-off runs.

## CLI usage

~~~bash
python3 scripts/transcribe_video_long.py /path/to/video.mp4 \
  --language auto \
  --whisper-model "$HOME/.local/share/whisper.cpp/ggml-large-v3-turbo-q5_0.bin" \
  --translate \
  --translation-model "$HOME/.cache/hy-mt/HY-MT1.5-1.8B-Q4_K_M.gguf"
~~~

For a known English video:

~~~bash
python3 scripts/transcribe_video_long.py /path/to/video.mp4 \
  --language en \
  --translate \
  --translation-model "$HOME/.cache/hy-mt/HY-MT1.5-1.8B-Q4_K_M.gguf"
~~~

Chinese videos usually do not need translation:

~~~bash
python3 scripts/transcribe_video_long.py /path/to/video.mp4 --language zh
~~~

## Output layout

Unless --output-dir is provided, outputs are written under the current working directory:

~~~text
.teaching-video-runtime/outputs/whisper-video-translator/<video-name>-<source-sha256-prefix>/
~~~

Default visible outputs:

~~~text
<video-name>.sentences.md
<video-name>.srt
manifest.json
~~~

Hidden resumable cache:

~~~text
.segments/
~~~

Audio cache is removed after a successful run unless --keep-audio-cache or --full is used.

## Validation

Validate an existing output directory:

~~~bash
python3 scripts/transcribe_video_long.py validate OUTPUT_DIR \
  --stem VIDEO_BASENAME \
  --expect-bilingual
~~~

By default, validation results are embedded in manifest.json.
Use --emit-validation-json if you need a standalone validation.json.

## Install as a Codex skill

Clone this repository directly into your Codex skills directory:

~~~bash
mkdir -p "$HOME/.codex/skills"
git clone https://github.com/yangshuli330/whisper-video-translator.git \
  "$HOME/.codex/skills/transcribe-video-long"
~~~

Then start a new Codex task so the skill list refreshes.

## URL inputs

This skill intentionally handles local video files only. If the user provides a URL, first download it with a dedicated video-downloader workflow, then pass the downloaded local video path to this script.

This keeps responsibilities clear:

- downloader: fetch the media
- transcriber: turn a local media file into bilingual transcript artifacts

## Notes

- The bilingual SRT keeps Whisper cue timing.
- The Markdown reading draft is sentence-level: it reconstructs text from cues, splits by punctuation, and estimates sentence timestamps from cue text positions.
- This is not word-level timestamp alignment.
- Translation quality depends on the local model, prompts, and how the source audio is segmented by Whisper.

## Limitations

- This project expects local command-line runtimes and local model files; it does not bundle ffmpeg, whisper.cpp, llama.cpp, Whisper models, or Hunyuan models.
- URL downloading is intentionally out of scope. Download media first, then pass a local video path to the script.
- Subtitle timing follows Whisper cue timing. The Markdown draft is sentence-level with approximate sentence timestamps, not word-level forced alignment.
- Translation quality depends on the local translation model and source segmentation.
- The current implementation has been smoke-tested on macOS; broader cross-platform packaging and installer scripts are future work.
