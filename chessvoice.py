"""Chess coach for camera glasses on a video call.

Watch mode reads the board from the call window and speaks the engine's move
into the call. Voice mode listens for spoken moves instead; call audio comes in
on one device and replies go out on another, so it never hears itself, and
"look" reads the board from the call window when the tracked game drifts.
"""

import argparse
import os
import queue
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "ggml-small-q5_1.bin")
VAD_MODEL = os.path.join(HERE, "models", "ggml-silero-v5.1.2.bin")
RATE = 16000

# Whisper is steered toward chess vocabulary; without this it writes "night"
# for knight and "before" for b4.
WHISPER_PROMPT = (
    "Chess moves: e4, d5, knight f3, bishop c4, queen takes d5, rook e1, "
    "castle short, castle long, ok, undo, look, new game white, new game black. "
    "At f3, fil c4, vezir d5, kale e1, sah g1, kisa rok, uzun rok, tamam, geri al, bak."
)

PIECE_WORDS = {
    chess.KNIGHT: ["knight", "night", "horse", "at", "knights"],
    chess.BISHOP: ["bishop", "fil"],
    chess.ROOK: ["rook", "rock", "kale"],
    chess.QUEEN: ["queen", "vezir"],
    chess.KING: ["king", "sah"],
    chess.PAWN: ["pawn", "piyon", "porn"],
}

NUMBER_WORDS = {
    "one": "1", "two": "2", "to": "2", "too": "2", "three": "3", "four": "4",
    "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "bir": "1", "iki": "2", "uc": "3", "dort": "4", "bes": "5", "alti": "6",
    "yedi": "7", "sekiz": "8",
}

LETTER_WORDS = {"bee": "b", "be": "b", "see": "c", "sea": "c", "dee": "d",
                "ef": "f", "eff": "f", "gee": "g", "aitch": "h",
                "ce": "c", "de": "d", "fe": "f", "ge": "g", "he": "h"}

PHRASES = {
    "en": {
        "voice": "Samantha",
        "ready": "Ready. You play {color}.",
        "white": "white", "black": "black",
        "play": "Play {move}.",
        "wait": "Your opponent's turn. Tell me their move.",
        "illegal": "I did not get a legal move. Say it again.",
        "ambiguous": "Which piece? {options}.",
        "mate": "Checkmate.",
        "draw": "Draw.",
        "undone": "Undone.",
        "looking": "Looking.",
        "no_board": "I could not read the board.",
        "no_window": "No call window.",
        "check": ", check",
        "takes": "takes",
        "short": "castle short", "long": "castle long",
        "names": {chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
                  chess.QUEEN: "queen", chess.KING: "king", chess.PAWN: "pawn"},
    },
    "tr": {
        "voice": "Yelda",
        "ready": "Hazir. {color} oynuyorsun.",
        "white": "beyaz", "black": "siyah",
        "play": "{move} oyna.",
        "wait": "Rakibin sirasi. Hamlesini soyle.",
        "illegal": "Gecerli bir hamle anlamadim. Tekrar soyle.",
        "ambiguous": "Hangi tas? {options}.",
        "mate": "Mat.",
        "draw": "Berabere.",
        "undone": "Geri alindi.",
        "looking": "Bakiyorum.",
        "no_board": "Tahtayi okuyamadim.",
        "no_window": "Arama penceresi yok.",
        "check": ", sah",
        "takes": "alir",
        "short": "kisa rok", "long": "uzun rok",
        "names": {chess.KNIGHT: "at", chess.BISHOP: "fil", chess.ROOK: "kale",
                  chess.QUEEN: "vezir", chess.KING: "sah", chess.PAWN: "piyon"},
    },
}


def fold(text):
    """Lowercase and strip diacritics so 'şah' and 'sah' match the same word."""
    text = unicodedata.normalize("NFKD", text.lower().replace("ı", "i"))
    return "".join(c for c in text if not unicodedata.combining(c))


def normalise(text):
    text = fold(text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    words = [NUMBER_WORDS.get(w, LETTER_WORDS.get(w, w)) for w in text.split()]
    text = " ".join(words)
    # "e 4", "e-4", "E. 4" all mean e4.
    return re.sub(r"\b([a-h])[\s-]*([1-8])\b", r"\1\2", text)


def has_word(text, words):
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def parse_command(raw):
    """Turn a transcript into (kind, payload). kind is a command name or 'move'."""
    text = normalise(raw)
    if not text.strip():
        return None, None
    if has_word(text, ["new", "yeni"]) and has_word(text, ["game", "oyun"]):
        if has_word(text, ["black", "siyah"]):
            return "new", chess.BLACK
        return "new", chess.WHITE
    if has_word(text, ["undo", "geri"]):
        return "undo", None
    if has_word(text, ["look", "bak", "scan"]):
        return "look", None
    if has_word(text, ["repeat", "again", "tekrar"]):
        return "repeat", None
    # "ok" and "yes" are left out: they are what whisper invents from noise.
    if re.fullmatch(r"\s*(done|played|i played it|tamam|oynadim)\s*", text):
        return "accept", None
    return "move", text


def match_move(board, text):
    """Return (moves, piece) for the legal moves the spoken text can mean."""
    legal = list(board.legal_moves)
    long_castle = has_word(text, ["long", "uzun", "queenside"]) or "o-o-o" in text
    if has_word(text, ["castle", "castles", "rok", "castling"]) or "o-o" in text:
        moves = [m for m in legal if board.is_castling(m)]
        if long_castle:
            moves = [m for m in moves if chess.square_file(m.to_square) == 2]
        elif has_word(text, ["short", "kisa", "kingside"]):
            moves = [m for m in moves if chess.square_file(m.to_square) == 6]
        return moves, chess.KING

    squares = [chess.parse_square(s) for s in re.findall(r"\b[a-h][1-8]\b", text)]
    # Knight last: its Turkish word "at" is also English filler ("pawn at e4").
    named = [p for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KING, chess.PAWN,
                         chess.KNIGHT) if has_word(text, PIECE_WORDS[p])]
    piece = named[0] if named else None
    if not squares:
        return [], piece

    if len(squares) >= 2:
        moves = [m for m in legal if m.from_square == squares[0] and m.to_square == squares[1]]
    else:
        moves = [m for m in legal if m.to_square == squares[0]]
        # "e takes d5": a lone file letter before the target names the pawn's file.
        lone = re.search(r"\b([a-h])\b\s+(takes|alir|x|captures)", text)
        if lone:
            f = ord(lone.group(1)) - ord("a")
            moves = [m for m in moves if chess.square_file(m.from_square) == f]
        if piece is not None:
            by_piece = [m for m in moves if board.piece_type_at(m.from_square) == piece]
            # "e8 queen" names the promotion piece, not the piece that moves.
            if by_piece or not any(m.promotion for m in moves):
                moves = by_piece
    if any(m.promotion for m in moves):
        promo = next((p for p in named if p not in (chess.PAWN, chess.KING)), chess.QUEEN)
        moves = [m for m in moves if m.promotion == promo]
    return moves, piece


def describe(board, move, lang):
    p = PHRASES[lang]
    if board.is_castling(move):
        text = p["short"] if chess.square_file(move.to_square) == 6 else p["long"]
    else:
        name = p["names"][board.piece_type_at(move.from_square)]
        target = chess.square_name(move.to_square)
        if board.is_capture(move):
            text = f"{name} {p['takes']} {target}"
        else:
            text = f"{name} {target}"
        if move.promotion:
            text += f" {p['names'][move.promotion]}"
    if board.gives_check(move):
        text += p["check"]
    # Spelled with a space so the voice says "e 4", not "eh-four" as one word.
    return re.sub(r"\b([a-h])([1-8])\b", r"\1 \2", text)


class Speaker:
    def __init__(self, lang, device, listener, echo_guard):
        self.voice = PHRASES[lang]["voice"]
        self.device = device
        self.listener = listener
        self.echo_guard = echo_guard

    def say(self, text):
        print(f"  >> {text}", flush=True)
        cmd = ["say", "-v", self.voice]
        if self.device:
            cmd += ["-a", self.device]
        if self.listener:
            self.listener.paused = True
        subprocess.run(cmd + [text], check=False)
        if self.listener:
            # The reply can come back from the glasses over the call after a
            # network round trip; hearing it would replay the suggested move.
            time.sleep(self.echo_guard)
            self.listener.drain()
            self.listener.paused = False


class Listener:
    """Energy-gated utterance capture: speech starts above the noise floor and
    ends after a pause, which is all a one-phrase command needs."""

    FRAME = 480  # 30 ms at 16 kHz

    def __init__(self, device):
        import numpy as np
        import sounddevice as sd
        self.np = np
        self.q = queue.Queue()
        self.paused = False
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                                     blocksize=self.FRAME, device=device,
                                     callback=self._cb)
        self.stream.start()

    def _cb(self, data, frames, t, status):
        if not self.paused:
            self.q.put(data.copy().reshape(-1))

    def drain(self):
        while not self.q.empty():
            self.q.get_nowait()

    def utterance(self):
        """Speech starts when 3 of the last 5 frames sit well above the noise
        floor; it ends after 0.7 s below it. A fixed threshold failed: the same
        voice measured 150-600 RMS through a laptop mic and far more over a call."""
        np = self.np
        floor, recent, pre, frames, silent = None, [], [], [], 0
        while True:
            chunk = self.q.get()
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            if floor is None:
                floor = max(rms, 10.0)
            loud = rms > max(floor * 4, 60)
            if not frames:
                if not loud:
                    floor = 0.97 * floor + 0.03 * max(rms, 10.0)
                recent = (recent + [loud])[-5:]
                pre = (pre + [chunk])[-12:]
                if sum(recent) >= 3:
                    frames, silent = list(pre), 0
                continue
            frames.append(chunk)
            silent = 0 if loud else silent + 1
            if silent > 23 or len(frames) > 8 * RATE // self.FRAME:
                return np.concatenate(frames)


def transcribe(samples, lang):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())
    try:
        return transcribe_file(path, lang)
    finally:
        os.remove(path)


def transcribe_file(path, lang):
    out = subprocess.run(
        # Silero VAD drops noise-only clips; without it whisper turns park
        # noise into "ok." and that would play the suggested move.
        ["whisper-cli", "-m", MODEL, "-f", path, "-l", lang, "-nt", "-np", "-sns",
         "--vad", "-vm", VAD_MODEL, "--prompt", WHISPER_PROMPT],
        capture_output=True, text=True, check=False)
    return out.stdout.strip()


CALL_WINDOW = r"instagram"


def call_window(pattern=CALL_WINDOW):
    """Largest on-screen window whose app or title matches; a call popup is
    usually the biggest one, the inbox tab behind it the second."""
    import Quartz
    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    best, area, title = None, 0, None
    for w in Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID):
        name = f"{w.get('kCGWindowOwnerName', '')} {w.get('kCGWindowName', '')}"
        if w.get("kCGWindowLayer", 0) != 0 or not re.search(pattern, name, re.I):
            continue
        b = w.get("kCGWindowBounds", {})
        a = b.get("Width", 0) * b.get("Height", 0)
        if a > area:
            best, area, title = w.get("kCGWindowNumber"), a, name
    if title != getattr(call_window, "last", None):
        print(f"  watching window: {title}", flush=True)
        call_window.last = title
    return best


def capture_frame(path, pattern=CALL_WINDOW):
    wid = call_window(pattern)
    if wid is None:
        return False
    subprocess.run(["screencapture", "-x", "-o", "-l", str(wid), path], check=False)
    return os.path.exists(path) and os.path.getsize(path) > 0


VIEW_PROMPT = (
    "Read the image file {image}. It is a frame from glasses worn by a chess player "
    "sitting at a physical board. Ignore any app interface, overlays and hands. "
    "Report the board exactly as it appears in the image: 8 rows from the far edge "
    "of the board to the near edge, each row 8 characters from left to right. "
    "Use . for an empty square, PNBRQK for white pieces, pnbrqk for black pieces. "
    "Also say which colour's pieces started on the near side (look at where the "
    "kings and most pieces of each colour sit). Answer with JSON only, no prose: "
    '{{"near": "white" or "black", "rows": ["8 chars", ... 8 rows]}}. '
    'If the whole board is not visible, answer {{"error": "reason"}}.')


def view_to_map(rows, near):
    """Turn the camera-view grid into {square: symbol}. The model only reports
    what it sees; doing the rotation here keeps a black-side view from being
    mirrored by a model that is bad at mental rotation."""
    if len(rows) != 8 or any(len(r) != 8 for r in rows):
        return None
    seen = {}
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            if c == ".":
                continue
            if c not in "PNBRQKpnbrqk":
                return None
            if near == chess.WHITE:
                sq = chess.square(j, 7 - i)
            else:
                sq = chess.square(7 - j, i)
            seen[sq] = c
    return seen


def shrink(image, longest=1024):
    """A 1024 px frame reads as well as the 1800 px original and uploads faster."""
    out = image + ".small.png"
    r = subprocess.run(["sips", "-Z", str(longest), image, "--out", out],
                       capture_output=True, check=False)
    return out if r.returncode == 0 and os.path.exists(out) else image


def read_view(image, model):
    """Returns (near_colour, {square: symbol}) or None."""
    import json
    small = shrink(image)
    # Plugins, MCP servers, hooks and settings are skipped: they cost five
    # seconds of start-up per read and a board read needs none of them.
    out = subprocess.run(
        ["claude", "-p", VIEW_PROMPT.format(image=small), "--model", model,
         "--tools", "Read", "--allowedTools=Read", "--strict-mcp-config",
         "--disable-slash-commands", "--no-chrome", "--no-session-persistence",
         "--setting-sources", ""],
        capture_output=True, text=True, check=False, timeout=120,
        cwd=tempfile.gettempdir())
    if small != image:
        os.remove(small)
    m = re.search(r"\{.*\}", out.stdout, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if "rows" not in data:
        return None
    near = chess.BLACK if str(data.get("near", "")).lower() == "black" else chess.WHITE
    seen = view_to_map([str(r) for r in data["rows"]], near)
    return (near, seen) if seen is not None else None


def placement_of(seen):
    board = chess.Board(None)
    for sq, c in seen.items():
        board.set_piece_at(sq, chess.Piece.from_symbol(c))
    return board.board_fen()


def diff(board, seen):
    """Squares where the board and the camera read disagree."""
    return sum(1 for sq in chess.SQUARES
               if (board.piece_at(sq).symbol() if board.piece_at(sq) else None) != seen.get(sq))


def infer_moves(board, seen, tolerance=2):
    """Find the move sequence (0 to 2 plies) that best explains what the camera
    sees. A misread square or two is forgiven because the answer must also be
    legal, which a raw read of the board cannot promise."""
    best, best_score = [], diff(board, seen)
    for m1 in board.legal_moves:
        board.push(m1)
        s1 = diff(board, seen)
        if s1 < best_score:
            best, best_score = [m1], s1
        if s1 > 0:
            for m2 in board.legal_moves:
                board.push(m2)
                s2 = diff(board, seen)
                # A second ply has to earn its place, or noise invents moves.
                if s2 < best_score - 1:
                    best, best_score = [m1, m2], s2
                board.pop()
        board.pop()
    return (best, best_score) if best_score <= tolerance else (None, best_score)


def board_from_placement(placement, to_move):
    board = chess.Board(None)
    board.set_board_fen(placement)
    board.turn = to_move
    rights = ""
    for color, rank, k, q in ((chess.WHITE, 0, "K", "Q"), (chess.BLACK, 7, "k", "q")):
        if board.piece_at(chess.square(4, rank)) == chess.Piece(chess.KING, color):
            if board.piece_at(chess.square(7, rank)) == chess.Piece(chess.ROOK, color):
                rights += k
            if board.piece_at(chess.square(0, rank)) == chess.Piece(chess.ROOK, color):
                rights += q
    board.set_castling_fen(rights or "-")
    return board if board.is_valid() else None


class Game:
    def __init__(self, args, speaker):
        self.args = args
        self.p = PHRASES[args.lang]
        self.speak = speaker.say
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        self.engine.configure({"Skill Level": args.skill})
        self.new(chess.WHITE if args.color == "white" else chess.BLACK)

    def new(self, color):
        self.me = color
        self.board = chess.Board()
        self.pending = None
        self.speak(self.p["ready"].format(color=self.p["white" if color else "black"]))
        self.next()

    def next(self):
        if self.board.is_checkmate():
            return self.speak(self.p["mate"])
        if self.board.is_game_over(claim_draw=True):
            return self.speak(self.p["draw"])
        if self.board.turn != self.me:
            self.pending = None
            return
        result = self.engine.play(self.board, chess.engine.Limit(time=self.args.think))
        self.pending = result.move
        self.speak(self.p["play"].format(move=describe(self.board, result.move, self.args.lang)))

    def handle(self, text):
        kind, payload = parse_command(text)
        if kind is None:
            return
        if kind == "new":
            return self.new(payload)
        if kind == "undo":
            if self.board.move_stack:
                self.board.pop()
            self.speak(self.p["undone"])
            return self.next()
        if kind == "repeat":
            if self.pending:
                return self.speak(self.p["play"].format(
                    move=describe(self.board, self.pending, self.args.lang)))
            return self.speak(self.p["wait"])
        if kind == "look":
            return self.look()
        if kind == "accept":
            if self.pending is None:
                return self.speak(self.p["wait"])
            self.board.push(self.pending)
            self.pending = None
            return self.next()
        # Park chatter and our own echoed "Play knight f6" must not become moves.
        castle = has_word(payload, ["castle", "castles", "rok"])
        if not (re.search(r"\b[a-h][1-8]\b", payload) or castle) or \
                has_word(payload, ["play", "oyna"]):
            print("  (ignored)", flush=True)
            return
        moves, _ = match_move(self.board, payload)
        if len(moves) == 1:
            self.board.push(moves[0])
            print(f"  board: {self.board.fen()}", flush=True)
            return self.next()
        if len(moves) > 1:
            opts = ", ".join(describe(self.board, m, self.args.lang) for m in moves[:3])
            return self.speak(self.p["ambiguous"].format(options=opts))
        self.speak(self.p["illegal"])

    def look(self):
        self.speak(self.p["looking"])
        path = os.path.join(tempfile.gettempdir(), "chessvoice-frame.png")
        if self.args.frame:
            path = self.args.frame
        elif not capture_frame(path, self.args.window):
            return self.speak(self.p["no_window"])
        try:
            view = read_view(path, self.args.vision_model)
        finally:
            if not self.args.frame and os.path.exists(path):
                os.remove(path)
        board = board_from_placement(placement_of(view[1]), self.me) if view else None
        if board is None:
            return self.speak(self.p["no_board"])
        print(f"  read: {board.fen()}", flush=True)
        self.board = board
        self.next()


class Watcher:
    """Watch mode: the camera is the only input. Every read is explained as
    legal moves from the tracked game; a move is committed once two reads in a
    row agree, so a hand over the board mid-move changes nothing."""

    START = chess.Board().board_fen()

    def __init__(self, args, speaker):
        self.args = args
        self.p = PHRASES[args.lang]
        self.speak = speaker.say
        self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        self.engine.configure({"Skill Level": args.skill})
        self.board, self.me = None, None
        self.candidate, self.misses, self.last_miss = None, 0, None

    def begin(self, near, seen):
        self.me = near
        if diff(chess.Board(), seen) <= 2:
            self.board = chess.Board()
        else:
            self.board = board_from_placement(placement_of(seen), near)
            if self.board is None:
                return False
        self.candidate, self.misses = None, 0
        self.speak(self.p["ready"].format(color=self.p["white" if near else "black"]))
        self.next()
        return True

    def next(self):
        if self.board.is_checkmate():
            return self.speak(self.p["mate"])
        if self.board.is_game_over(claim_draw=True):
            return self.speak(self.p["draw"])
        if self.board.turn == self.me:
            move = self.engine.play(self.board, chess.engine.Limit(time=self.args.think)).move
            self.speak(self.p["play"].format(move=describe(self.board, move, self.args.lang)))

    def step(self, view):
        near, seen = view
        if self.board is None:
            return self.begin(near, seen)
        # Pieces set up again: a new game, and the player may have switched sides.
        if len(self.board.move_stack) > 4 and diff(chess.Board(), seen) <= 2:
            return self.begin(near, seen)
        moves, score = infer_moves(self.board, seen)
        if moves is None:
            placement = placement_of(seen)
            self.misses += 1
            print(f"  no legal explanation ({score} squares off), miss {self.misses}", flush=True)
            # The game has drifted beyond repair: trust the camera, but only a
            # read that repeats, since one bad read must not rewrite the game.
            if self.misses >= 4 and placement == self.last_miss:
                board = board_from_placement(placement, self.me)
                if board is not None:
                    print(f"  resync: {board.fen()}", flush=True)
                    self.board, self.misses = board, 0
                    self.next()
            self.last_miss = placement
            return
        self.misses = 0
        if not moves:
            self.candidate = None
            return
        key = [m.uci() for m in moves]
        # A read that matches a legal position square for square is trusted at
        # once; only a read with misread squares waits for a second opinion.
        if key != self.candidate and score > 0:
            self.candidate = key
            print(f"  saw {' '.join(key)}, waiting for a second read", flush=True)
            return
        self.candidate = None
        for m in moves:
            self.board.push(m)
        print(f"  moved {' '.join(key)}: {self.board.fen()}", flush=True)
        self.next()


def pin_call_input(device):
    """Make the reply device the system input, so the call app hears the coach.
    macOS falls back to the built-in microphone after a restart or a device
    change, and the call then carries room noise instead of the moves."""
    tool = os.path.join(HERE, "tools", "setinput")
    if not os.path.exists(tool):
        subprocess.run(["swiftc", "-O", tool + ".swift", "-o", tool], check=False)
    subprocess.run([tool, device], check=False)


def watch(args, speaker):
    """Two reads are kept in flight, started half a read apart, so the board
    is looked at twice as often as one read takes. Results are applied in the
    order the frames were taken; a frame older than one already applied is
    dropped."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    if args.out_device:
        pin_call_input(args.out_device)
    w = Watcher(args, speaker)
    lock = threading.Lock()
    state = {"applied": 0.0, "seq": 0}

    def one(taken, path):
        t0 = time.time()
        try:
            view = read_view(path, args.vision_model)
        finally:
            if not args.frame and os.path.exists(path):
                os.remove(path)
        with lock:
            print(f"read in {time.time() - t0:.1f}s", flush=True)
            if taken < state["applied"]:
                return
            state["applied"] = taken
            if view is None:
                print("  board not readable", flush=True)
            else:
                w.step(view)

    pool = ThreadPoolExecutor(max_workers=args.parallel)
    try:
        while True:
            state["seq"] += 1
            path = args.frame or os.path.join(
                tempfile.gettempdir(), f"chessvoice-watch-{state['seq']}.png")
            if not args.frame and not capture_frame(path, args.window):
                print("  no call window", flush=True)
                time.sleep(3)
                continue
            pool.submit(one, time.time(), path)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        w.engine.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--color", choices=["white", "black"], default="white")
    ap.add_argument("--lang", choices=["en", "tr"], default="en")
    ap.add_argument("--in-device", help="call audio, e.g. 'BlackHole 16ch'")
    ap.add_argument("--out-device", help="reply audio, e.g. 'BlackHole 2ch'")
    ap.add_argument("--echo-guard", type=float, default=1.5,
                    help="seconds of deafness after speaking")
    ap.add_argument("--think", type=float, default=1.0, help="engine seconds per move")
    ap.add_argument("--skill", type=int, default=20, help="Stockfish skill 0-20")
    ap.add_argument("--vision-model", default="sonnet")
    ap.add_argument("--frame", help="read the board from this image instead of the call")
    ap.add_argument("--window", default=CALL_WINDOW,
                    help="regex for the call window's app or title")
    ap.add_argument("--watch", action="store_true",
                    help="read moves from the camera; no listening")
    ap.add_argument("--interval", type=float, default=3.0,
                    help="watch mode: seconds between frames sent for reading")
    ap.add_argument("--parallel", type=int, default=2,
                    help="watch mode: board reads in flight at once")
    ap.add_argument("--type", action="store_true", help="type commands instead of speaking")
    args = ap.parse_args()

    if args.watch:
        return watch(args, Speaker(args.lang, args.out_device, None, 0))
    listener = None if args.type else Listener(args.in_device)
    # Same device in and out means the assistant would transcribe its own voice.
    speaker = Speaker(args.lang, args.out_device, listener, args.echo_guard)
    game = Game(args, speaker)
    try:
        while True:
            if args.type:
                text = input("> ")
            else:
                text = transcribe(listener.utterance(), args.lang)
                print(f"heard: {text}", flush=True)
            game.handle(text)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        game.engine.quit()


if __name__ == "__main__":
    sys.exit(main())
