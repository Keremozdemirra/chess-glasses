import subprocess
import tempfile
import os

import chess

from chessvoice import match_move, normalise, parse_command, describe, transcribe_file


def one(board, text):
    moves, _ = match_move(board, normalise(text))
    assert len(moves) == 1, (text, moves)
    return moves[0].uci()


def test_spoken_variants():
    b = chess.Board()
    assert one(b, "e4") == "e2e4"
    assert one(b, "E. 4") == "e2e4"
    assert one(b, "knight to f three") == "g1f3"
    assert one(b, "Night F3") == "g1f3"
    assert one(b, "at f3") == "g1f3"
    assert one(b, "pawn at e4") == "e2e4"
    assert one(b, "piyon d4") == "d2d4"


def test_capture_and_file():
    b = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/2P1P3/8/PP1P1PPP/RNBQKBNR w KQkq - 0 2")
    assert one(b, "e takes d5") == "e4d5"
    assert one(b, "c takes d5") == "c4d5"
    moves, _ = match_move(b, normalise("takes d5"))
    assert len(moves) == 2


def test_castling_and_promotion():
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert one(b, "castle short") == "e1g1"
    assert one(b, "uzun rok") == "e1c1"
    b = chess.Board("8/4P3/8/8/8/8/k7/7K w - - 0 1")
    assert one(b, "e8 queen") == "e7e8q"
    assert one(b, "pawn e8 knight") == "e7e8n"


def test_commands_turkish_without_diacritics():
    assert parse_command("Yeni oyun siyah") == ("new", chess.BLACK)
    assert parse_command("tamam") == ("accept", None)
    assert parse_command("geri al")[0] == "undo"
    assert parse_command("bak")[0] == "look"
    assert parse_command("Şah g1")[0] == "move"


def test_describe_reads_squares_apart():
    b = chess.Board()
    assert describe(b, chess.Move.from_uci("g1f3"), "en") == "knight f 3"


def test_speech_round_trip():
    """The real pipeline end to end: synthesised voice through whisper.cpp."""
    d = tempfile.mkdtemp()
    for phrase, uci in (("knight f3", "g1f3"), ("e4", "e2e4"), ("pawn d4", "d2d4")):
        aiff, wav = os.path.join(d, "a.aiff"), os.path.join(d, "a.wav")
        subprocess.run(["say", "-v", "Samantha", "-o", aiff, phrase], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", aiff, "-ar", "16000",
                        "-ac", "1", wav], check=True)
        heard = transcribe_file(wav, "en")
        assert one(chess.Board(), heard) == uci, (phrase, heard)


def _feed(samples):
    """Run samples through the real utterance gate without an audio device."""
    import queue
    import numpy as np
    from chessvoice import Listener
    L = Listener.__new__(Listener)
    L.np, L.q, L.paused = np, queue.Queue(), False
    for i in range(0, len(samples) - Listener.FRAME, Listener.FRAME):
        L.q.put(samples[i:i + Listener.FRAME])
    return L.utterance()


def _voice(phrase):
    import numpy as np
    d = tempfile.mkdtemp()
    aiff, raw = os.path.join(d, "a.aiff"), os.path.join(d, "a.raw")
    subprocess.run(["say", "-v", "Samantha", "-o", aiff, phrase], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", aiff, "-ar", "16000", "-ac", "1",
                    "-f", "s16le", raw], check=True)
    return np.fromfile(raw, dtype=np.int16)


def test_gate_and_vad_in_noise():
    import numpy as np
    from chessvoice import transcribe
    rng = np.random.default_rng(1)
    noise = lambda n: rng.normal(0, 150, n).astype(np.int16)
    # 1 s of noise, the move, 2 s of noise: the gate must cut out the move.
    voice = _voice("knight f3")
    clip = np.concatenate([noise(16000), voice + noise(len(voice)), noise(32000)])
    heard = transcribe(_feed(clip), "en")
    assert one(chess.Board(), heard) == "g1f3", heard
    # A loud burst of noise opens the gate; it must not come out as a command.
    burst = np.concatenate([noise(16000), noise(24000) * 8, noise(32000)]).astype(np.int16)
    kind, _ = parse_command(transcribe(_feed(burst), "en"))
    assert kind in (None, "move"), kind
