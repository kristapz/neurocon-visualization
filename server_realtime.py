#!/usr/bin/env python3
"""
server_realtime.py - Fixed version with correct file serving
"""

import asyncio
import os
import sys
import json
import numpy as np
import nibabel as nib
from typing import Dict, List
from datetime import datetime
from pathlib import Path
from collections import deque

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

# Configuration
CONTEXT_WINDOW_SEC = 45
MIN_WORDS_FOR_BRAIN = 30
MAX_BRAIN_INTERVAL = 60
SR = 16000
UTTER_MS_MIN = 2500
UTTER_MS_MAX = 8000
SILENCE_RMS = 0.002
SILENCE_MS_END = 700

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve HTML
@app.get("/")
async def root():
    return FileResponse("live_brain.html")


# Serve brain template explicitly
@app.get("/mni152_t1_2mm.nii.gz")
async def serve_mni():
    file_path = "mni152_t1_2mm.nii.gz"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/gzip")
    else:
        # Try visualization/static path
        alt_path = "visualization/static/mni152_t1_2mm.nii.gz"
        if os.path.exists(alt_path):
            return FileResponse(alt_path, media_type="application/gzip")
    return {"error": "MNI template not found"}


# Mount static directories - be explicit about paths
if os.path.exists("visualization/static"):
    app.mount("/static", StaticFiles(directory="visualization/static"), name="static")

if os.path.exists("output"):
    app.mount("/output", StaticFiles(directory="output"), name="output")

# Global state
clients = set()
brain_processor = None
current_session = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    try:
        # Send session info if available
        if current_session:
            await websocket.send_bytes(orjson.dumps({
                "kind": "session_info",
                "session_id": current_session["id"],
                "start_time": current_session["start_time"]
            }))

        # Keep connection alive
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
        "word_count": 0
    }


class TextAccumulator:
    """Manages text buffering for brain processing"""

    def __init__(self):
        self.accumulated_text = []
        self.word_count = 0
        self.last_brain_time = asyncio.get_event_loop().time()

    def add_text(self, text: str):
        self.accumulated_text.append(text)
        self.word_count += len(text.split())

    def should_process(self) -> bool:
        current_time = asyncio.get_event_loop().time()
        time_elapsed = current_time - self.last_brain_time

        if self.word_count >= MIN_WORDS_FOR_BRAIN:
            return True
        if time_elapsed >= MAX_BRAIN_INTERVAL and self.word_count > 10:
            return True
        return False

    def get_text(self) -> str:
        return " ".join(self.accumulated_text)

    def reset(self):
        self.word_count = 0
        self.last_brain_time = asyncio.get_event_loop().time()
        self.accumulated_text = []


async def process_brain_map(text: str, session: Dict):
    """Generate and save brain map"""
    try:
        print(f"[BRAIN] Processing: {text[:50]}...")

        # Generate brain map
        loop = asyncio.get_event_loop()
        brain_img, similarity, idx = await loop.run_in_executor(
            None,
            brain_processor.text_to_brain_map,
            text,
            0.15
        )

        # Save brain map
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:19]  # Include milliseconds

        # Save to output/maps
        map_filename = f"brain_{timestamp_str}.nii.gz"
        map_path = f"output/maps/{map_filename}"
        nib.save(brain_img, map_path)

        # Also save to session directory
        session_map_path = session["path"] / map_filename
        nib.save(brain_img, str(session_map_path))

        # Update session
        map_info = {
            "file": map_path,
            "timestamp": timestamp.isoformat(),
            "similarity": float(similarity),
            "index": int(idx),
            "text_preview": text[:200]
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
            "map_file": f"/{map_path}",  # Leading slash for absolute path
            "similarity": float(similarity),
            "index": int(idx),
            "text": text[:200],
            "timestamp": timestamp.isoformat(),
            "session_id": session["id"]
        })

        print(f"[BRAIN] Saved: {map_path} (similarity: {similarity:.3f})")

    except Exception as e:
        print(f"[BRAIN] Error: {e}")
        import traceback
        traceback.print_exc()


async def audio_processing_loop():
    """Main audio → text → brain loop"""
    global brain_processor, current_session

    # Initialize session
    current_session = create_session()
    print(f"[SESSION] Started: {current_session['id']}")

    # Initialize components
    print("[BRAIN] Initializing TextToBrainViz...")
    brain_processor = TextToBrainViz()
    print("[BRAIN] Ready!")

    print("[ASR] Initializing Whisper...")
    asr = StreamingASR(model_size="base.en", device="auto", compute_type="int8")
    print("[ASR] Ready!")

    # Text accumulator
    accumulator = TextAccumulator()

    # Processing state
    seq = 0
    loop = asyncio.get_event_loop()

    # Audio buffers
    buf = []
    buf_samples = 0
    since_voice_ms = 0

    print("[AUDIO] Starting capture...")

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

        if text:
            seq += 1
            await broadcast(msg_final_sentence(seq, text, s0, s1, words))

            # Add to session
            current_session["transcript"].append(text)
            current_session["word_count"] += len(text.split())

            # Add to accumulator
            accumulator.add_text(text)

            # Check if should process brain
            if accumulator.should_process():
                context_text = accumulator.get_text()

                # Process brain map asynchronously
                asyncio.create_task(process_brain_map(context_text, current_session))

                accumulator.reset()


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    # Ensure directories exist
    Path("output/maps").mkdir(parents=True, exist_ok=True)
    Path("output/sessions").mkdir(parents=True, exist_ok=True)

    # Start audio processing
    asyncio.create_task(audio_processing_loop())
    print("[SERVER] Ready at http://localhost:8001")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)