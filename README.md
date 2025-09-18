# Speech2Brain: Real-Time Speech-to-Brain Visualization System

## Overview

Speech2Brain is a real-time system that converts spoken language into 3D brain activation maps, built on the NeuroConText framework for contrastive text-to-brain mapping.

**Original Paper**: [NeuroConText: Contrastive Text-to-Brain Mapping for Neuroscientific Literature](https://link.springer.com/chapter/10.1007/978-3-031-72384-1_31)

**Extended Version**: [bioRxiv Preprint](https://www.biorxiv.org/content/10.1101/2025.05.23.655707v1.abstract)

**Supplementary Material**: [Google Drive](https://drive.google.com/file/d/17IJ7Jn9cHXbMiBEzCnTepDcleeXHpRN-/view?usp=drive_link)

## Features

- Real-time speech transcription using Whisper
- Live 3D brain visualization with smooth morphing transitions
- Text-to-brain mapping using pre-trained NeuroConText model
- DiFuMo-512 functional brain parcellation
- WebSocket-based streaming architecture

## Prerequisites

- Python 3.8-3.10 (tested with 3.10.9)
- 16GB RAM minimum (32GB recommended)
- 10GB free storage
- Microphone for speech input
- Modern web browser with WebGL support

## Installation

### 1. Clone Repository

```bash
git clone [repository-url]
cd Speech2Brain
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv .venv310
source .venv310/bin/activate  # On Windows: .venv310\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper transformers nibabel nilearn
pip install fastapi uvicorn websockets orjson sounddevice
```

### 4. Download Data

Download the pre-trained embeddings from [Zenodo](https://zenodo.org/records/14169410?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImM0NmUwMWZhLWVmYzQtNDUxZS05NTg3LWJjZDdhZGY5MGRiYyIsImRhdGEiOnt9LCJyYW5kb20iOiI3MDlhYjYwYWYwN2Q1Y2JmYWU0MjE0NTFlNGYzMTQxZiJ9.p7EhGnpNIBN73FOn-L5MmQ9Dz5Cx86Y9x7kZWUyVz_fTp_lLxEEb21c4aBC-wb9Fbyg7dF8r1uHycu2I_dZBXw) and save in the `data` folder.

## Directory Structure

```
Speech2Brain/
├── data/                          # Pre-trained embeddings (~8GB)
│   ├── preprocessed_train_text_embeddings.pkl
│   └── preprocessed_train_gaussian_embeddings.pkl
│
├── src/                           # Core modules
│   ├── embeddings.py
│   └── [other research modules]
│
├── Speech Components/             # Audio/ASR pipeline
│   ├── asr_worker.py             # Whisper ASR
│   ├── audio_stream.py           # Microphone capture
│   └── ws_protocol.py            # WebSocket protocol
│
├── visualization/                 # Web interface
│   └── static/
│       └── niivue.umd.js        # 3D viewer
│
├── output/                        # Generated outputs
│   ├── maps/                     # Brain activation maps
│   └── sessions/                 # Session recordings
│
├── server_realtime.py            # Main server
├── texttobrain.py               # Text-to-brain pipeline
├── live_brain.html              # Web interface
├── layers.py                    # Neural network definitions
├── losses.py                    # Loss functions
├── main.py                      # Training script
├── best_val.pt                  # Model weights
└── mni152_t1_2mm.nii.gz        # Brain template
```

## Usage

### Real-Time Speech-to-Brain

1. **Start the server**:
```bash
python server_realtime.py
```

2. **Open browser**:
Navigate to `http://localhost:8001`

3. **Allow microphone access** and start speaking

### Training Models

For model training and evaluation:
```bash
python main.py
```

## Configuration

Edit `server_realtime.py`:

```python
# Audio processing
UTTER_MS_MIN = 1000      # Min utterance length
UTTER_MS_MAX = 5000      # Max utterance length
SILENCE_MS_END = 400     # Silence threshold

# Brain visualization  
DIFUMO_SMOOTHING = 0.7   # Transition smoothing (0-1)
```

## ASR Models

In `asr_worker.py`:
- `tiny.en`: Fastest, real-time
- `base.en`: Better accuracy
- `small.en`: Best balance
- `medium.en`: Highest accuracy

## GPU Acceleration

For NVIDIA GPUs:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Output Files

Each session generates:
```
output/sessions/session_TIMESTAMP/
├── transcript.txt       # Full transcript
├── metadata.json        # Scores and timing
└── brain_*.nii.gz      # Brain maps
```

## Troubleshooting

**No microphone input**: Check permissions and run `python -m sounddevice` to list devices

**Slow performance**: Use `tiny.en` model, increase cache size, enable GPU

**Missing embeddings**: Ensure `data/*.pkl` files are downloaded from Zenodo

## Citation

If you use this system, please cite:

```bibtex
@inproceedings{ghayem2024neurocontext,
  title={NeuroConText: Contrastive Text-to-Brain Mapping for Neuroscientific Literature},
  author={Ghayem, Fateme and others},
  booktitle={MICCAI},
  year={2024}
}
```

## License

Supported by KARAIB AI chair (ANR-20-CHIA-0025-01), ANR-22-PESN-0012 France 2030 program, and HORIZON-INFRA-2022-SERV-B-01 EBRAINS 2.0.

## Contact

For questions: fateme[dot]ghayem[at]gmail[dot]com