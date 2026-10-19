#!/bin/sh
# One-time setup on macOS. BlackHole installs an audio driver, so Homebrew
# will ask for your password.
set -e
cd "$(dirname "$0")"

brew install stockfish whisper-cpp ffmpeg
brew install --cask blackhole-2ch blackhole-16ch
swiftc -O tools/setinput.swift -o tools/setinput

python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

mkdir -p models
fetch() {
  [ -f "models/$1" ] || curl -fL -o "models/$1" "$2"
  echo "$3  models/$1" | shasum -a 256 -c -
}
fetch ggml-small-q5_1.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin \
  ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb
fetch ggml-silero-v5.1.2.bin \
  https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin \
  29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf

echo "Done. Run: .venv/bin/python chessvoice.py --in-device 'BlackHole 16ch' --out-device 'BlackHole 2ch'"
