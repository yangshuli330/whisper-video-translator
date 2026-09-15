#!/usr/bin/env bash
set -euo pipefail

echo "whisper-video-translator macOS installer"
echo

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required but was not found."
  echo "Install Homebrew first: https://brew.sh/"
  exit 1
fi

install_if_missing() {
  local formula="$1"
  local command_name="$2"
  if command -v "$command_name" >/dev/null 2>&1; then
    echo "[OK] $command_name"
    return
  fi
  echo "[INSTALL] brew install $formula"
  brew install "$formula"
}

install_if_missing ffmpeg ffmpeg
install_if_missing whisper-cpp whisper-cli
install_if_missing llama.cpp llama-completion

echo
echo "Command-line dependencies are installed or already available."
echo
echo "Next steps:"
echo "1. Download or prepare a whisper.cpp-compatible model."
echo "   Default expected path:"
echo "   $HOME/.local/share/whisper.cpp/ggml-large-v3-turbo-q5_0.bin"
echo
echo "2. Optional: prepare a GGUF translation model, for example Hunyuan-MT."
echo
echo "3. Run:"
echo "   python3 scripts/transcribe_video_long.py doctor"
echo
echo "You can also configure model paths with:"
echo "   export WVT_WHISPER_MODEL=/path/to/whisper-model.bin"
echo "   export WVT_TRANSLATION_MODEL=/path/to/translation-model.gguf"
