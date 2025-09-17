# audio_stream.py
import os, queue
import numpy as np, sounddevice as sd
from typing import Iterator, Tuple

SR = 16000
FRAME_S = float(os.environ.get("FRAME_S", "0.64"))
HOP_S   = float(os.environ.get("HOP_S",   "0.32"))
FRAME = int(SR * FRAME_S)
HOP   = int(SR * HOP_S)

def _select_device():
    dev = os.environ.get("AUDIO_DEVICE_INDEX")
    if dev is not None:
        idx = int(dev)
        sd.default.device = (idx, idx)
        print(f"[AUDIO] using device index {idx}")
    print(f"[AUDIO] default device: {sd.default.device}")

def mic_chunker() -> Iterator[Tuple[np.ndarray, float]]:
    """
    Yield float32 frames of length FRAME with 50 percent overlap, and a running end timestamp (seconds).
    """
    _select_device()
    q = queue.Queue()
    t_samples = 0

    def callback(indata, frames, time, status):
        if status:
            # status may contain overflow/underflow flags; you can print if debugging
            pass
        q.put(bytes(indata))  # raw bytes

    # RawInputStream is strict about dtype and channels
    stream = sd.RawInputStream(
        samplerate=SR,
        blocksize=HOP,
        channels=1,
        dtype="float32",
        callback=callback,
    )
    stream.start()

    ring = np.zeros((FRAME, 1), dtype=np.float32)
    filled = 0
    while True:
        block = q.get()
        if block is None:
            break
        block = np.frombuffer(block, dtype=np.float32).reshape(-1, 1)
        if block.shape[0] != HOP:
            # some backends may deliver a different size; just skip odd blocks
            continue
        ring[:-HOP] = ring[HOP:]
        ring[-HOP:] = block
        filled = min(FRAME, filled + HOP)
        t_samples += HOP
        if filled == FRAME:
            t1 = t_samples / SR
            yield ring[:, 0].copy(), t1

def rms_level(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-8))
