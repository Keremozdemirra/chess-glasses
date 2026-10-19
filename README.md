# chess-glasses

A voice chess coach for camera glasses without a display. You wear the glasses, call your Mac on WhatsApp, say your opponent's move, and Stockfish's reply is spoken into your ear.

```
glasses mic ──WhatsApp call──▶ Mac ──▶ whisper.cpp ──▶ move parser ──▶ Stockfish
glasses ear ◀──WhatsApp call── Mac ◀── macOS say ◀──────────────────────┘
glasses camera ──video──▶ WhatsApp window ──"look"──▶ Claude reads the board (optional)
```

Speech recognition and the engine run on the Mac. Two BlackHole virtual audio devices keep the directions apart, so the coach never hears its own voice.

## Setup (macOS, Apple silicon)

```sh
./setup.sh
```

It installs Stockfish, whisper.cpp, ffmpeg and the two BlackHole drivers with Homebrew, creates a Python venv, and downloads two models from Hugging Face, checked by SHA-256: whisper `small` (q5_1) and the Silero voice-activity model.

In WhatsApp on the Mac, open Settings, then Calls:

| Setting | Device |
|---|---|
| Microphone | BlackHole 2ch |
| Speaker | BlackHole 16ch |

The glasses call from your phone's WhatsApp account, so the Mac needs a second account to answer. An account cannot call itself.

## Run

```sh
.venv/bin/python chessvoice.py --color white \
  --in-device "BlackHole 16ch" --out-device "BlackHole 2ch"
```

Without the device flags it uses the Mac's own microphone and speakers, which is handy for a dry run. `--type` takes commands from the keyboard.

## What to say

| Say | Effect |
|---|---|
| `e5`, `knight f6`, `bishop takes c4`, `e takes d5`, `castle short` | the move just played on the board |
| `done` | you played the move the coach suggested |
| `undo` | take back the last move |
| `repeat` | hear the suggestion again |
| `look` | read the board from the WhatsApp video window and resync |
| `new game black` | start over as black |

Commands also work in Turkish without diacritics (`tamam`, `geri al`, `bak`, `yeni oyun siyah`) with `--lang tr`. Moves are recognised far more reliably in English (`--lang en`).

Anything without a square or a castling word is ignored, so table talk does not move pieces. After every reply the input is muted for 1.5 s (`--echo-guard`) so the reply, echoed back over the call, is not taken for a move. `ok` and `yes` do not confirm a move: whisper produces them from background noise.

## What leaves the Mac

- The call itself goes through WhatsApp.
- `look` sends one screenshot of the WhatsApp window to Claude through the `claude` CLI. Anything else visible in that window goes with it. The screenshot is deleted afterwards.
- Audio is transcribed locally and each clip is deleted after transcription.

## Tests

```sh
.venv/bin/python -m pytest -q
```

They cover spoken move variants, captures, castling, promotion, the Turkish commands, and the audio path end to end: synthesised speech mixed with noise passes through the voice gate and whisper and must come out as the right move, while a burst of noise must not come out as a command.

Not yet tested: a live call from the glasses, and `look` on a clear frame of a full board.

## Fair play

Engine help during a rated, online or tournament game is cheating and gets accounts banned. Use this for training, analysis, or games where your opponent knows.

## Licence

GPL-3.0, because it builds on python-chess and Stockfish, both GPL-3.0.
