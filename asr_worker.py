# asr_worker.py - Optimized for speed
from faster_whisper import WhisperModel
import numpy as np
from typing import Dict, Any, List, Tuple

class StreamingASR:
    def __init__(self, model_size="tiny.en", device="auto", compute_type="int8"):
        """
        Initialize Whisper model optimized for speed

        Model sizes (speed vs accuracy tradeoff):
        - tiny.en: ~39M, fastest, good enough for conversation
        - base.en: ~74M, better accuracy, still fast
        - small.en: ~244M, good balance
        - medium.en: ~769M, high accuracy but slow
        """

        # Auto-detect CUDA availability
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"  # Use fp16 on GPU
                    print(f"[ASR] Using CUDA GPU acceleration")
                else:
                    device = "cpu"
                    print(f"[ASR] Using CPU (no GPU detected)")
            except ImportError:
                device = "cpu"
                print(f"[ASR] Using CPU (torch not available)")

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

        print(f"[ASR] Loaded {model_size} model on {device} with {compute_type}")

    def transcribe_window(self, audio: np.ndarray) -> Tuple[List[Dict[str, float]], str, float, float]:
        """
        Transcribe audio window with minimal latency

        Returns:
            words: List of word dictionaries with timestamps
            text: Combined text string
            t0: Start time
            t1: End time
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

    def transcribe_batch(self, audio_batch: List[np.ndarray]) -> List[Tuple]:
        """
        Batch transcription for efficiency (if needed)
        Note: faster-whisper doesn't natively support batching,
        but we can process sequentially with shared model
        """
        results = []
        for audio in audio_batch:
            results.append(self.transcribe_window(audio))
        return results