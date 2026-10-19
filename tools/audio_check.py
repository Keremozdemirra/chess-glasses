"""Checks that speech sent to an output device comes back on its input side.
Prints the peak level; 0 means the loopback passes nothing."""

import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd

device = sys.argv[1] if len(sys.argv) > 1 else "BlackHole 2ch"
threading.Thread(target=lambda: (time.sleep(0.5),
                                 subprocess.run(["say", "-a", device, "test"]))).start()
x = sd.rec(int(2.5 * 16000), samplerate=16000, channels=1, dtype="int16", device=device)
sd.wait()
print(f"{device}: peak {int(np.abs(x).max())}")
