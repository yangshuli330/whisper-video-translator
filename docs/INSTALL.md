# Installation

This project is intentionally local-first. It does not bundle external binaries or model weights.

## macOS quick install

~~~bash
./scripts/install_macos.sh
python3 scripts/transcribe_video_long.py doctor
~~~

The installer checks or installs:

- ffmpeg / ffprobe
- whisper-cli from whisper.cpp
- llama-completion from llama.cpp

It does not download model weights.

## Required tools

You can install dependencies manually if preferred:

~~~bash
brew install ffmpeg whisper-cpp llama.cpp
~~~

Then run:

~~~bash
python3 scripts/transcribe_video_long.py doctor
~~~

## Model files

You need a whisper.cpp-compatible ASR model.

Default path:

~~~text
~/.local/share/whisper.cpp/ggml-large-v3-turbo-q5_0.bin
~~~

You can override it with:

~~~bash
export WVT_WHISPER_MODEL="/path/to/whisper-model.bin"
~~~

or pass it directly:

~~~bash
python3 scripts/transcribe_video_long.py video.mp4 --whisper-model "/path/to/whisper-model.bin"
~~~

For translation, prepare a GGUF model that can be loaded by llama-completion.

Example environment variable:

~~~bash
export WVT_TRANSLATION_MODEL="/path/to/HY-MT1.5-1.8B-Q4_K_M.gguf"
~~~

Then verify:

~~~bash
python3 scripts/transcribe_video_long.py doctor \
  --translation-model "$WVT_TRANSLATION_MODEL"
~~~

## Output root

Default output root:

~~~text
.teaching-video-runtime/outputs/whisper-video-translator/
~~~

Override it with:

~~~bash
export WVT_OUTPUT_ROOT="/path/to/transcript-outputs"
~~~

or per run with:

~~~bash
python3 scripts/transcribe_video_long.py video.mp4 -o /path/to/output
~~~

## Common issues

### whisper-cli not found

Install whisper.cpp and ensure the binary is on PATH.

### llama-completion not found

Translation requires llama.cpp. If you only need original-language SRT, omit --translate.

### model does not exist

Run doctor with explicit paths:

~~~bash
python3 scripts/transcribe_video_long.py doctor \
  --whisper-model "/path/to/whisper-model.bin" \
  --translation-model "/path/to/translation-model.gguf"
~~~

### URL input

Download the video first with your preferred downloader. This project processes local video files.
