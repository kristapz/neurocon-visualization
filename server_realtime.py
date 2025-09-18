#!/usr/bin/env python3
"""
server_realtime.py - Optimized for M1 Mac with streaming embeddings
"""

import asyncio
import os
import sys
import json
import numpy as np
import nibabel as nib
import torch
import hashlib
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from collections import deque
from threading import Semaphore

import orjson
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add Speech Components to path
sys.path.append("Speech Components")

# Import components
from texttobrain import TextToBrainViz
from asr_worker import StreamingASR
from audio_stream import mic_chunker, rms_level
from ws_protocol import msg_partial, msg_final_sentence, msg_metrics

# ===== CONFIGURATION =====
# ASR Settings - Optimized for M1
UTTER_MS_MIN = 1000  # 1 second minimum
UTTER_MS_MAX = 5000  # 5 second max
SILENCE_RMS = 0.003
SILENCE_MS_END = 400  # 400ms silence
SR = 16000

# Brain Processing Settings
MIN_WORDS_PER_CHUNK = 0
MIN_CHUNKS_FOR_BRAIN = 1
MAX_CHUNKS_FOR_BRAIN = 10
MAX_CONCURRENT_EMBEDS = 3  # Reduced for M1
EMBEDDING_CACHE_SIZE = 500

# Smoothing Settings
DIFUMO_SMOOTHING = 0.7
USE_SMOOTHING = True

# ===== CONCURRENCY CONTROL =====
embedding_semaphore = Semaphore(MAX_CONCURRENT_EMBEDS)
embedding_tasks = []


# ===== TEXT CLEANING =====
def clean_repetitive_text(text: str, max_repeats: int = 3) -> str:
    """Remove excessive word repetitions from text"""
    words = text.split()
    if not words:
        return text

    cleaned = []
    prev_word = None
    repeat_count = 0

    for word in words:
        if word.lower() == prev_word:
            repeat_count += 1
            if repeat_count < max_repeats:
                cleaned.append(word)
        else:
            cleaned.append(word)
            prev_word = word.lower()
            repeat_count = 0

    return " ".join(cleaned)


# ===== EXTENDED TextToBrainViz - M1 OPTIMIZED =====
class OptimizedTextToBrainViz(TextToBrainViz):
    """Extended version with M1-specific optimizations"""

    def __init__(self):
        super().__init__()
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.running_coeffs = None
        self.smoothing_factor = 0.7

        # Pre-compute numpy versions for faster similarity on M1
        self.train_brain_latents_np = self.train_brain_latents.cpu().numpy()

        # Pre-cache common phrases at startup
        common_phrases = [
            "hello", "thinking about", "I feel", "what if",
            "let me", "I think", "actually", "basically",
            "you know", "so", "um", "uh"
        ]
        print("[CACHE] Pre-computing common phrases...")
        for phrase in common_phrases:
            _ = self.text_to_embedding_cached(phrase)
        print(f"[CACHE] Pre-cached {len(common_phrases)} phrases")

    def smooth_difumo_coefficients(self, new_coeffs):
        """Apply exponential moving average to coefficients for smooth transitions"""
        if self.running_coeffs is None:
            self.running_coeffs = new_coeffs.copy()
        else:
            # Exponential moving average
            self.running_coeffs = (self.running_coeffs * self.smoothing_factor +
                                   new_coeffs * (1 - self.smoothing_factor))
        return self.running_coeffs

    def text_to_embedding_cached(self, text):
        """Cached version of text_to_embedding_4096"""
        # Generate cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Check cache
        if text_hash in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[text_hash]

        # Compute embedding
        self.cache_misses += 1
        embedding = self.text_to_embedding_4096(text)

        # Update cache with LRU eviction
        if len(self.embedding_cache) >= EMBEDDING_CACHE_SIZE:
            # Remove oldest entry
            oldest = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest]

        self.embedding_cache[text_hash] = embedding
        return embedding

    def embedding_to_brain_map(self, embedding_4096, top_pct=0.15, use_smoothing=True):
        """Generate brain map from pre-computed embedding - M1 optimized"""
        # Project to latent space
        with torch.no_grad():
            text_latent = self.model.encode_text(
                torch.from_numpy(embedding_4096).float()
            )

        # Use numpy for similarity computation (faster on M1)
        text_latent_np = text_latent.cpu().numpy().squeeze()

        # Normalize for cosine similarity
        text_latent_np = text_latent_np / np.linalg.norm(text_latent_np)
        train_norms = np.linalg.norm(self.train_brain_latents_np, axis=1, keepdims=True)
        normalized_train = self.train_brain_latents_np / train_norms

        # Compute similarities using numpy
        similarities = np.dot(normalized_train, text_latent_np)

        best_idx = similarities.argmax()
        best_score = similarities[best_idx]

        # Get DiFuMo coefficients
        difumo_coeffs = self.train_gaussian_embeddings[best_idx]

        # Apply smoothing if enabled
        if use_smoothing:
            difumo_coeffs = self.smooth_difumo_coefficients(difumo_coeffs)

        # Reconstruct brain volume from coefficients
        brain_img = self.masker.inverse_transform(difumo_coeffs.reshape(1, -1))

        # Apply threshold
        brain_img_sparse = self.top_percent_threshold(brain_img, top_pct=top_pct)

        return brain_img_sparse, float(best_score), int(best_idx)


# ===== STREAMING ACCUMULATOR =====
class StreamingAccumulator:
    """Manages streaming embeddings with concurrency control"""

    def __init__(self, brain_processor):
        self.brain_processor = brain_processor
        self.chunk_embeddings = deque(maxlen=MAX_CHUNKS_FOR_BRAIN)
        self.chunk_texts = deque(maxlen=MAX_CHUNKS_FOR_BRAIN)
        self.total_words = 0
        self.chunks_processed = 0
        self.last_brain_time = asyncio.get_event_loop().time()

    async def add_chunk_async(self, text: str, loop):
        """Add and immediately embed a text chunk"""
        words = text.split()
        word_count = len(words)

        # Skip empty chunks only
        if word_count == 0:
            return False

        self.total_words += word_count
        self.chunk_texts.append(text)

        # Embed with concurrency control
        async def embed_with_limit():
            with embedding_semaphore:
                embedding = await loop.run_in_executor(
                    None,
                    self.brain_processor.text_to_embedding_cached,
                    text
                )
                return embedding

        try:
            # Create embedding task
            task = asyncio.create_task(embed_with_limit())
            embedding_tasks.append(task)

            # Wait for embedding
            embedding = await task
            self.chunk_embeddings.append(embedding)
            self.chunks_processed += 1

            print(
                f"[EMBED] Chunk {self.chunks_processed}: {word_count} words "
                f"(cache: {self.brain_processor.cache_hits}/{self.brain_processor.cache_misses})"
            )
            return True

        except Exception as e:
            print(f"[EMBED] Error: {e}")
            return False
        finally:
            # Clean up completed tasks
            embedding_tasks[:] = [t for t in embedding_tasks if not t.done()]

    def should_generate_brain(self) -> bool:
        """Check if we have enough data for brain generation"""
        return len(self.chunk_embeddings) > 0

    def get_combined_embedding(self) -> Optional[np.ndarray]:
        """Combine chunk embeddings using weighted average"""
        if not self.chunk_embeddings:
            return None

        # Convert to numpy array
        embeddings = list(self.chunk_embeddings)

        # Weight recent chunks more heavily
        weights = np.linspace(0.5, 1.0, len(embeddings))
        weights = weights / weights.sum()

        # Weighted average
        combined = np.zeros_like(embeddings[0])
        for emb, w in zip(embeddings, weights):
            combined += emb * w

        return combined

    def get_context_text(self) -> str:
        """Get combined text for display"""
        return " ".join(self.chunk_texts)

    def reset_for_next_brain(self):
        """Complete reset after brain generation"""
        self.chunk_embeddings.clear()
        self.chunk_texts.clear()
        self.last_brain_time = asyncio.get_event_loop().time()
        self.total_words = 0


# ===== FASTAPI SETUP =====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return FileResponse("live_brain.html")


@app.get("/mni152_t1_2mm.nii.gz")
async def serve_mni():
    file_path = "mni152_t1_2mm.nii.gz"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/gzip")
    return {"error": "MNI template not found"}


# Mount static directories
if os.path.exists("visualization/static"):
    app.mount("/static", StaticFiles(directory="visualization/static"), name="static")
if os.path.exists("output"):
    app.mount("/output", StaticFiles(directory="output"), name="output")

# Global state
clients = set()
brain_processor = None
current_session = None


# ===== WEBSOCKET =====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    try:
        if current_session:
            await websocket.send_bytes(orjson.dumps({
                "kind": "session_info",
                "session_id": current_session["id"],
                "start_time": current_session["start_time"]
            }))

        while True:
            await asyncio.sleep(60)

    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception as e:
        print(f"[WS] Error: {e}")
        clients.discard(websocket)


async def broadcast(message: Dict):
    """Broadcast to all connected clients"""
    if not clients:
        return

    data = orjson.dumps(message)
    disconnected = []

    for client in list(clients):
        try:
            await client.send_bytes(data)
        except:
            disconnected.append(client)

    for client in disconnected:
        clients.discard(client)


# ===== SESSION MANAGEMENT =====
def create_session():
    """Create new session"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{timestamp}"
    session_path = Path(f"output/sessions/{session_id}")
    session_path.mkdir(parents=True, exist_ok=True)

    return {
        "id": session_id,
        "path": session_path,
        "start_time": datetime.now().isoformat(),
        "transcript": [],
        "brain_maps": [],
        "word_count": 0,
        "chunk_count": 0
    }


# ===== BRAIN GENERATION =====
async def generate_brain_from_embedding(embedding: np.ndarray, context_text: str, session: Dict):
    """Generate brain map from pre-computed embedding"""
    try:
        print(f"[BRAIN] Generating from combined embedding...")

        # Generate brain map from embedding
        loop = asyncio.get_event_loop()
        brain_img, similarity, idx = await loop.run_in_executor(
            None,
            brain_processor.embedding_to_brain_map,
            embedding,
            0.15,
            USE_SMOOTHING
        )

        # Save brain map
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:19]

        map_filename = f"brain_{timestamp_str}.nii.gz"
        map_path = f"output/maps/{map_filename}"
        nib.save(brain_img, map_path)

        # Save to session
        session_map_path = session["path"] / map_filename
        nib.save(brain_img, str(session_map_path))

        # Update metadata
        map_info = {
            "file": map_path,
            "timestamp": timestamp.isoformat(),
            "similarity": float(similarity),
            "index": int(idx),
            "text_preview": context_text[:200],
            "chunk_count": session["chunk_count"]
        }
        session["brain_maps"].append(map_info)

        # Update index.json
        with open("index.json", "w") as f:
            json.dump({
                "files": [m["file"] for m in session["brain_maps"]],
                "metadata": session["brain_maps"],
                "session_id": session["id"]
            }, f, indent=2)

        # Broadcast update
        await broadcast({
            "kind": "brain_update",
            "map_file": f"/{map_path}",
            "similarity": float(similarity),
            "index": int(idx),
            "text": context_text[:200],
            "timestamp": timestamp.isoformat(),
            "session_id": session["id"],
            "chunk_count": session["chunk_count"]
        })

        print(f"[BRAIN] Generated: {map_path} (similarity: {similarity:.3f})")

    except Exception as e:
        print(f"[BRAIN] Error: {e}")
        import traceback
        traceback.print_exc()


# ===== MAIN PROCESSING LOOP =====
async def audio_processing_loop():
    """Main audio → text → embedding → brain loop"""
    global brain_processor, current_session

    # Initialize session
    current_session = create_session()
    print(f"[SESSION] Started: {current_session['id']}")

    # Initialize M1-optimized brain processor
    print("[BRAIN] Initializing OptimizedTextToBrainViz (M1 optimized)...")
    brain_processor = OptimizedTextToBrainViz()
    brain_processor.smoothing_factor = DIFUMO_SMOOTHING
    print(f"[BRAIN] Ready! (Smoothing: {DIFUMO_SMOOTHING}, Model: GPT-Neo-125M)")

    print("[ASR] Initializing Whisper (tiny model for M1)...")
    # Force tiny model for M1 performance
    asr = StreamingASR(model_size="tiny.en", device="cpu", compute_type="int8")
    print("[ASR] Ready!")

    # Streaming accumulator
    accumulator = StreamingAccumulator(brain_processor)

    # Processing state
    seq = 0
    loop = asyncio.get_event_loop()

    # Audio buffers
    buf = []
    buf_samples = 0
    since_voice_ms = 0

    print("[AUDIO] Starting capture...")
    print(f"[CONFIG] Min chunk: {MIN_WORDS_PER_CHUNK} words, Max concurrent: {MAX_CONCURRENT_EMBEDS}")

    # Main loop
    for audio, _t1 in mic_chunker():
        # Calculate RMS
        rms = float(rms_level(audio))
        await broadcast(msg_metrics(rms=rms))

        # Accumulate audio
        buf.append(audio.astype("float32", copy=False))
        buf_samples += audio.shape[0]
        elapsed_ms = int(1000 * buf_samples / SR)

        # Track silence
        if rms > SILENCE_RMS:
            since_voice_ms = 0
        else:
            since_voice_ms += int(1000 * audio.shape[0] / SR)

        # Check if should decode
        should_decode = False
        if elapsed_ms >= UTTER_MS_MIN and since_voice_ms >= SILENCE_MS_END:
            should_decode = True
        if elapsed_ms >= UTTER_MS_MAX:
            should_decode = True

        if not should_decode:
            continue

        # Process audio
        utter = np.concatenate(buf, axis=0)
        buf.clear()
        buf_samples = 0
        since_voice_ms = 0

        # Run ASR
        try:
            words, text, s0, s1 = await loop.run_in_executor(
                None, asr.transcribe_window, utter.copy()
            )
        except Exception as e:
            print(f"[ASR] Error: {e}")
            continue

        # Send updates
        if words:
            seq += 1
            await broadcast(msg_partial(seq, words))
            print(f"[ASR] {len(words)} words")

        if text:  # Remove word count check - embed everything
            # Clean repetitive text
            text = clean_repetitive_text(text, max_repeats=3)

            seq += 1
            await broadcast(msg_final_sentence(seq, text, s0, s1, words))

            # Add to session
            current_session["transcript"].append(text)
            current_session["word_count"] += len(text.split())
            current_session["chunk_count"] += 1

            # Save transcript periodically
            if current_session["chunk_count"] % 10 == 0:
                transcript_path = current_session["path"] / "transcript.txt"
                with open(transcript_path, "w") as f:
                    f.write("\n".join(current_session["transcript"]))

            # Stream embedding immediately
            embedded = await accumulator.add_chunk_async(text, loop)

            if embedded:
                # Broadcast embedding status
                await broadcast({
                    "kind": "embedding_status",
                    "chunks_embedded": accumulator.chunks_processed,
                    "total_words": accumulator.total_words,
                    "cache_hits": brain_processor.cache_hits,
                    "cache_misses": brain_processor.cache_misses
                })

            # Check if should generate brain
            if accumulator.should_generate_brain():
                combined_embedding = accumulator.get_combined_embedding()

                if combined_embedding is not None:
                    context_text = accumulator.get_context_text()

                    # Generate brain map asynchronously
                    asyncio.create_task(
                        generate_brain_from_embedding(
                            combined_embedding,
                            context_text,
                            current_session
                        )
                    )

                    # Partial reset keeping context
                    accumulator.reset_for_next_brain()


# ===== STARTUP =====
@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    Path("output/maps").mkdir(parents=True, exist_ok=True)
    Path("output/sessions").mkdir(parents=True, exist_ok=True)

    # Clean up old index.json if exists
    if os.path.exists("index.json"):
        os.remove("index.json")

    asyncio.create_task(audio_processing_loop())
    print("[SERVER] Ready at http://localhost:8001")
    print("[SERVER] M1 optimizations enabled: GPT-Neo-125M, numpy similarity, int8 Whisper")


# ===== CLEANUP =====
@app.on_event("shutdown")
async def shutdown():
    """Clean up pending tasks and save session"""
    # Cancel pending embedding tasks
    for task in embedding_tasks:
        if not task.done():
            task.cancel()

    # Save final session data
    if current_session:
        session_path = current_session["path"]

        # Save final transcript
        transcript_path = session_path / "transcript.txt"
        with open(transcript_path, "w") as f:
            f.write("\n".join(current_session["transcript"]))

        # Save metadata
        metadata_path = session_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "id": current_session["id"],
                "start_time": current_session["start_time"],
                "end_time": datetime.now().isoformat(),
                "word_count": current_session["word_count"],
                "chunk_count": current_session["chunk_count"],
                "brain_maps_generated": len(current_session["brain_maps"])
            }, f, indent=2)

    print(f"[SERVER] Shutdown complete")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)