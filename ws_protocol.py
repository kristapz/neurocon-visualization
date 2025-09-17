"""
ws_protocol.py - WebSocket message protocol definitions
Defines message formats for real-time communication
"""

from typing import List, Dict, Any

def msg_partial(seq: int, words: List[Dict[str, float]]) -> Dict[str, Any]:
    """Word-by-word transcription updates"""
    return {
        "kind": "partial",
        "seq": seq,
        "words": words
    }

def msg_final_sentence(
    seq: int,
    text: str,
    t0: float,
    t1: float,
    words: List[Dict[str, float]]
) -> Dict[str, Any]:
    """Complete sentence with timing"""
    return {
        "kind": "final_sentence",
        "seq": seq,
        "text": text,
        "t0": t0,
        "t1": t1,
        "words": words
    }

def msg_metrics(
    rms: float,
    pitch: float = None,
    tempo: float = None
) -> Dict[str, Any]:
    """Audio metrics for visualization"""
    return {
        "kind": "metrics",
        "rms": rms,
        "pitch": pitch,
        "tempo": tempo
    }

def msg_brain_update(
    map_file: str,
    similarity: float,
    index: int,
    text: str,
    timestamp: str,
    session_id: str
) -> Dict[str, Any]:
    """Brain map generation update"""
    return {
        "kind": "brain_update",
        "map_file": map_file,
        "similarity": similarity,
        "index": index,
        "text": text,
        "timestamp": timestamp,
        "session_id": session_id
    }

def msg_session_info(
    session_id: str,
    start_time: str
) -> Dict[str, Any]:
    """Session metadata"""
    return {
        "kind": "session_info",
        "session_id": session_id,
        "start_time": start_time
    }

def msg_processing_status(
    is_processing: bool,
    stage: str = None
) -> Dict[str, Any]:
    """Processing status updates"""
    return {
        "kind": "processing_status",
        "is_processing": is_processing,
        "stage": stage
    }