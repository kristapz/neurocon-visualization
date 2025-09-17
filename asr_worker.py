# asr_worker.py
from faster_whisper import WhisperModel
import numpy as np
from typing import Dict, Any, List, Tuple

class StreamingASR:
    def __init__(self, model_size="base.en", device="auto", compute_type="int8"):
        # CPU int8 for your current wheel
        self.model = WhisperModel("medium.en", device="cpu", compute_type="int8")
        self.decode_kwargs = dict(
            language="en",
            # we already do our own silence endpointing, keep VAD conservative or off
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=True,   # larger utterances, let decoder use context
            beam_size=5,                       # small beam for stability
            patience=1.05,                     # must be > 0 with beam search
            temperature=0.0,
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            # Optional domain hint
            # initial_prompt="Neuroscience, embeddings, hippocampus, thalamus, pulvinar, Broca, Wernicke, transformer."
        )

    def transcribe_window(self, audio: np.ndarray) -> Tuple[List[Dict[str, float]], str, float, float]:
        segs, _ = self.model.transcribe(audio=audio, **self.decode_kwargs)
        words, text, t0, t1 = [], "", None, None
        for s in segs:
            if t0 is None: t0 = s.start
            t1 = s.end
            if s.words:
                for w in s.words:
                    words.append({"w": w.word, "t0": float(w.start), "t1": float(w.end)})
            text += s.text
        return words, text.strip(), float(t0 or 0.0), float(t1 or 0.0)
