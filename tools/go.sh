#!/bin/sh
# Starts a session from nothing: keeps the display awake for as long as the
# coach runs (a sleeping display stops the browser drawing the call), brings
# Chrome forward so the call window is drawn, then runs watch mode.
cd "$(dirname "$0")/.." || exit 1
caffeinate -dims -w $$ &
osascript -e 'tell application "Google Chrome" to activate'
exec .venv/bin/python chessvoice.py --watch --out-device "BlackHole 2ch" "$@"
