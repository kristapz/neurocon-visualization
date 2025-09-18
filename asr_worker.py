# asr_worker.py - Optimized for M1 Mac
from faster_whisper import WhisperModel
import numpy as np
from typing import Dict, Any, List, Tuple


class StreamingASR:
    def __init__(self, model_size="tiny.en", device="auto", compute_type="int8"):
        """
        Initialize Whisper model optimized for M1 Mac
        Force tiny model and int8 for best performance
        """

        # Force tiny model for M1
        model_size = "tiny.en"

        # M1 optimization: Use CPU with int8
        # (Metal not yet supported by faster-whisper)
        device = "cpu"
        compute_type = "int8"
        print(f"[ASR] Using {model_size} on CPU with int8 (optimized for M1)")

        # Load model
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # Optimized decode settings for speed
        self.decode_kwargs = dict(
            language="en",
            # Speed optimizations
            beam_size=1,  # Greedy search (fastest)
            best_of=1,  # No sampling
            temperature=0.0,  # Deterministic
            # Quality settings
            vad_filter=False,  # We handle VAD ourselves
            word_timestamps=True,  # Need for alignment
            condition_on_previous_text=False,  # Faster without context
            # Thresholds
            no_speech_threshold=0.5,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )

        print(f"[ASR] Loaded {model_size} model optimized for M1")

    def transcribe_window(self, audio: np.ndarray) -> Tuple[List[Dict[str, float]], str, float, float]:
        """
        Transcribe audio window with minimal latency
        """
        # Quick check for silence (avoid processing empty audio)
        if np.max(np.abs(audio)) < 0.001:
            return [], "", 0.0, 0.0

        # Transcribe
        segments, info = self.model.transcribe(
            audio=audio,
            **self.decode_kwargs
        )

        # Process segments
        words = []
        text_parts = []
        t0, t1 = None, None

        for segment in segments:
            if t0 is None:
                t0 = segment.start
            t1 = segment.end

            # Add text
            text_parts.append(segment.text.strip())

            # Add word timestamps if available
            if segment.words:
                for word in segment.words:
                    words.append({
                        "w": word.word,
                        "t0": float(word.start),
                        "t1": float(word.end)
                    })

        # Combine text
        text = " ".join(text_parts).strip()

        return words, text, float(t0 or 0.0), float(t1 or 0.0)