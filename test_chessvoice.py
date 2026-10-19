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


# --- watch mode: moves inferred from camera reads ---

from chessvoice import infer_moves, view_to_map, Watcher


def seen_of(board):
    return {sq: p.symbol() for sq, p in board.piece_map().items()}


def rows_as_seen(board, near):
    """What a player on the `near` side sees: far edge first, left to right."""
    rows = []
    for i in range(8):
        row = ""
        for j in range(8):
            sq = chess.square(j, 7 - i) if near == chess.WHITE else chess.square(7 - j, i)
            p = board.piece_at(sq)
            row += p.symbol() if p else "."
        rows.append(row)
    return rows


def test_view_rotation_for_both_sides():
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    for near in (chess.WHITE, chess.BLACK):
        assert view_to_map(rows_as_seen(b, near), near) == seen_of(b)
    # From black's chair white's king is on the far row, fourth from the left.
    assert rows_as_seen(chess.Board(), chess.BLACK)[0] == "RNBKQBNR"


def test_infers_one_move_through_two_misread_squares():
    b = chess.Board()
    after = b.copy()
    after.push_san("e4")
    seen = seen_of(after)
    seen[chess.A7] = "P"          # wrong colour
    seen.pop(chess.H2)            # hidden by a hand
    moves, score = infer_moves(b, seen)
    assert [m.uci() for m in moves] == ["e2e4"] and score == 2


def test_infers_both_plies_when_a_read_was_missed():
    b = chess.Board()
    after = b.copy()
    for san in ("e4", "c5"):
        after.push_san(san)
    moves, _ = infer_moves(b, seen_of(after))
    assert [m.uci() for m in moves] == ["e2e4", "c7c5"]


def test_castling_and_capture_are_recognised():
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    after = b.copy()
    after.push_san("O-O")
    assert [m.uci() for m in infer_moves(b, seen_of(after))[0]] == ["e1g1"]
    b = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    after = b.copy()
    after.push_san("exd5")
    assert [m.uci() for m in infer_moves(b, seen_of(after))[0]] == ["e4d5"]


def test_garbage_read_is_not_a_move():
    b = chess.Board()
    seen = seen_of(b)
    for sq in (chess.D2, chess.E2, chess.F2, chess.D7, chess.E7):
        seen.pop(sq)                # a hand across the middle of the board
    moves, _ = infer_moves(b, seen)
    assert moves is None


class _Args:
    lang, skill, think = "en", 20, 0.2


def _watcher():
    said = []

    class Quiet:
        def say(self, t):
            said.append(t)
    return Watcher(_Args(), Quiet()), said


def test_watcher_commits_exact_read_at_once():
    w, said = _watcher()
    w.step((chess.BLACK, seen_of(chess.Board())))
    assert w.me == chess.BLACK and said == ["Ready. You play black."]
    after = chess.Board()
    after.push_san("d4")
    w.step((chess.BLACK, seen_of(after)))
    assert [m.uci() for m in w.board.move_stack] == ["d2d4"]
    assert said[-1].startswith("Play ")
    w.engine.quit()


def test_watcher_waits_for_second_read_when_squares_are_misread():
    w, _ = _watcher()
    w.step((chess.BLACK, seen_of(chess.Board())))
    after = chess.Board()
    after.push_san("d4")
    noisy = seen_of(after)
    noisy.pop(chess.H2)                               # hidden by a hand
    w.step((chess.BLACK, noisy))
    assert not w.board.move_stack                     # one noisy read is not enough
    w.step((chess.BLACK, noisy))
    assert [m.uci() for m in w.board.move_stack] == ["d2d4"]
    w.engine.quit()


def test_watcher_suggests_at_once_when_player_is_white():
    w, said = _watcher()
    w.step((chess.WHITE, seen_of(chess.Board())))
    assert said[0] == "Ready. You play white." and said[1].startswith("Play ")
    w.engine.quit()


def test_watcher_resyncs_only_on_a_repeated_bad_read():
    w, _ = _watcher()
    w.step((chess.WHITE, seen_of(chess.Board())))
    far = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5")
    for _ in range(3):
        w.step((chess.WHITE, seen_of(far)))
    assert not w.board.move_stack and w.board.board_fen() == chess.Board().board_fen()
    w.step((chess.WHITE, seen_of(far)))
    assert w.board.board_fen() == far.board_fen()
    w.engine.quit()
