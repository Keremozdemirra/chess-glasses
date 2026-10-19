# chess-glasses

A chess coach for camera glasses without a display. The glasses call a Mac at home over Instagram, the Mac reads the board from the call video, and Stockfish's move is spoken into your ear.

```
glasses camera ──Instagram video call──▶ Mac (Chrome) ──screenshot──▶ Claude reads the board
                                                                        │
                                              legal-move matcher ◀──────┘
                                                      │
glasses ear ◀──Instagram call── BlackHole mic ◀── say ◀── Stockfish
```

Only the Mac needs to be on. The phone stays in your pocket.

## What you need

- Camera glasses paired with your phone (tested with Ray-Ban Meta).
- A Mac with Homebrew, Python 3.12 or newer, the Xcode command line tools (`xcode-select --install`) and the `claude` CLI signed in. Tested on macOS 26.6 on Apple silicon with Python 3.14.
- A second Instagram account for the Mac. Instagram allows several accounts per person and signs them up with an email, so no second phone number is needed. One account cannot call itself.

## Setup

```sh
./setup.sh
```

It installs Stockfish, whisper.cpp, ffmpeg and the BlackHole 2ch virtual audio driver, creates a Python venv and downloads two speech models checked by SHA-256. BlackHole asks for your Mac password. If BlackHole passes silence afterwards, restart the Mac.

## Play

1. On the Mac, open instagram.com in **Google Chrome** and sign in with the second account. Safari puts its own echo cancellation on the microphone and sends silence from a virtual one.
2. Start the coach:

   ```sh
   tools/go.sh
   ```

   It keeps the display awake while the coach runs, brings Chrome forward and starts watch mode (`chessvoice.py --watch --out-device "BlackHole 2ch"`). Extra flags are passed through.

   It sets the Mac's input to BlackHole, unmuted, before anything else. macOS falls back to the built-in microphone after a restart. The first run asks for Screen Recording permission for your terminal app; allow it in System Settings, Privacy & Security, then start the coach again. Without it the captured frames are blank.
3. From your phone's Instagram, video call the second account and answer on the Mac. If the call screen says another microphone is connected, open its gear menu and pick BlackHole 2ch.
4. Double-press the capture button on the glasses so the call shows their camera.
5. Look at the whole board for a few seconds. You hear which colour you play and, when it is your turn, the move.

Keep the call window visible on the Mac and the Mac unlocked. A minimised or covered browser window, or a locked screen, stops drawing video.

## How a move is read

Every few seconds the call window is captured and Claude reports the board as the camera sees it, far row first, plus which colour sits nearest. The rotation to board squares happens in code, since a model asked to rotate a black-side view tends to mirror it.

A read is then explained as zero, one or two legal moves from the tracked game, forgiving up to two misread squares. A read that matches a legal position square for square is accepted at once; one with misread squares counts only when the next read agrees, so a hand over the board changes nothing. After four reads that no legal move explains, and only if two of them match, the coach trusts the camera and resyncs. Setting the pieces up again starts a new game.

A frame goes out every 3 seconds with two reads in flight. In the live test on 2026-10-19 one read took 9 to 17 seconds, so a clean read announces a move within roughly that time, and a blurry one waits for the next.

## Voice mode

Without `--watch` the coach listens instead of looking (for listening to a call, also install `blackhole-16ch` and pass `--in-device "BlackHole 16ch"`): say the opponent's move (`e5`, `knight f6`, `castle short`), say `done` after playing the suggestion. Speech is transcribed on the Mac by whisper.cpp with Silero voice detection. Run `.venv/bin/python chessvoice.py -h` for the flags.

## What leaves the Mac

- The call goes through Instagram.
- Each board read sends one screenshot of the call window to Claude through the `claude` CLI. The file is deleted after the read.
- Speech in voice mode is transcribed locally.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `no call window` | The call is in a background tab. Give it its own window. |
| Coach speaks, you hear nothing | The call is not using BlackHole, or the call was started before the input changed. Start a new call. |
| `tools/audio_check.py` prints peak 0 | BlackHole is muted. Running the coach unmutes it; a fresh install may need a restart. |
| You hear nothing from the Mac at all | The phone or glasses side: try a new call and the glasses volume swipe. |

## Tests

```sh
.venv/bin/python -m pytest -q
```

They cover the camera-view rotation for both sides, move inference through misread and hidden squares, both plies of a missed read, castling and captures, the two-read confirmation and the resync rule, plus the voice path end to end with synthesised speech in noise.

## Fair play

Engine help in rated, online or tournament games is cheating, and chess sites detect it automatically whatever your opponent agreed to. Use this for training, analysis, or casual games where your opponent knows.

## Licence

GPL-3.0, because it builds on python-chess and Stockfish, both GPL-3.0.
