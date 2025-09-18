Here's an updated README for your NeuroConText visualization project:

```markdown
# NeuroConText: Real-Time Speech-to-Brain Visualization

A real-time system that visualizes brain activation patterns from spoken language, built on the NeuroConText framework for contrastive text-to-brain mapping.

## Overview

This system converts speech to 3D brain activation maps in real-time by:
1. Transcribing speech using OpenAI's Whisper
2. Embedding text using language models (GPT-Neo)
3. Mapping embeddings to brain activation patterns via contrastive learning
4. Visualizing results as interactive 3D brain maps

Based on the paper: [NeuroConText: Contrastive Text-to-Brain Mapping for Neuroscientific Literature (MICCAI 2024)](https://github.com/ghayem/NeuroConText)

## Features

- 🎤 Real-time speech transcription with word-level timestamps
- 🧠 Live 3D brain visualization with smooth morphing transitions
- ⚡ Optimized for M1 Mac with Metal GPU acceleration
- 💾 Intelligent caching for repeated phrases
- 📊 DiFuMo-512 functional brain parcellation
- 🔄 WebSocket-based streaming architecture

## Prerequisites

- Python 3.8-3.10 (tested with 3.10)
- macOS with M1/M2 chip (Intel Macs also supported)
- 16GB RAM minimum (32GB recommended)
- 10GB free storage
- Microphone for speech input
- Modern web browser with WebGL support

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/kristapz/neurocon-visualization.git
cd neurocon-visualization
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv .venv310
source .venv310/bin/activate  # On Windows: .venv310\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install faster-whisper transformers nibabel nilearn
pip install fastapi uvicorn websockets orjson sounddevice
```

### 4. Download Pre-trained Data
Download the pre-trained embeddings (~8GB) from [Zenodo](https://zenodo.org/record/12789281) and save in the `data` folder:
- `preprocessed_train_text_embeddings.pkl`
- `preprocessed_train_gaussian_embeddings.pkl`

## Usage

### Real-Time Speech-to-Brain Visualization

1. Start the server:
```bash
python server_realtime.py
```

2. Open browser and navigate to: http://localhost:8001

3. Allow microphone access and start speaking!

### Interactive Mode (Text Input)

For text-based brain mapping without speech:
```bash
python texttobrain.py
```

## Configuration

### Performance Settings (server_realtime.py)

```python
# Audio processing
UTTER_MS_MIN = 1000      # Min utterance length (ms)
UTTER_MS_MAX = 5000      # Max utterance length (ms)
SILENCE_MS_END = 400     # Silence threshold (ms)

# Brain visualization  
DIFUMO_SMOOTHING = 0.7   # Transition smoothing (0-1)
MAX_CONCURRENT_EMBEDS = 3  # Parallel embedding limit
```

### Model Selection

The system is optimized for M1 Macs using:
- **Language Model**: GPT-Neo-125M (10x smaller than original, faster on M1)
- **ASR Model**: Whisper tiny.en with int8 quantization
- **Similarity Search**: NumPy-based (optimized for Apple Silicon)

To use a different language model, modify in `texttobrain.py`:
```python
model_name = "EleutherAI/gpt-neo-125M"  # Options: 125M, 1.3B, 2.7B
```

## Output Files

Each session generates:

```
output/
├── sessions/
│   └── session_TIMESTAMP/
│       ├── transcript.txt       # Full transcript
│       ├── metadata.json        # Session metadata
│       └── brain_*.nii.gz      # Brain maps
└── maps/
    └── brain_*.nii.gz          # All generated brain maps
```

## Architecture

The system combines several key components:

1. **Contrastive Model**: CLIP-style architecture mapping text to brain patterns
2. **Training Data**: ~20K neuroscience articles with activation coordinates
3. **DiFuMo Atlas**: Reduces brain volumes to 512 functional components
4. **Streaming Pipeline**: Real-time audio → ASR → embedding → brain mapping

### Key Optimizations for M1

- Smaller GPT-Neo-125M model (vs 1.3B original)
- NumPy-based similarity search (faster than PyTorch on M1)
- Metal GPU acceleration where supported
- Aggressive caching with pre-computed common phrases
- Zero-padding instead of duplication for embeddings

## Troubleshooting

**No microphone input**: 
```bash
python -m sounddevice  # List available devices
```

**Slow performance**: 
- Ensure you're using GPT-Neo-125M (not 1.3B)
- Check that Metal acceleration is enabled
- Reduce `MAX_CONCURRENT_EMBEDS` if needed

**Missing data files**: 
Download from [Zenodo](https://zenodo.org/record/12789281) and place in `data/` folder

**MPS errors on M1**: 
The system will automatically fall back to CPU if Metal operations fail

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

This project extends the original NeuroConText work and is provided for research purposes. The original work was supported by:
- KARAIB AI chair (ANR-20-CHIA-0025-01)
- ANR-22-PESN-0012 France 2030 program
- HORIZON-INFRA-2022-SERV-B-01 EBRAINS 2.0

## Contact

For questions about this visualization system: [your-email]

For questions about the original NeuroConText: fateme[dot]ghayem[at]gmail[dot]com
```

This README:
- Clearly explains what the system does
- Highlights M1 optimizations
- Provides clear installation and usage instructions
- Documents the configuration options
- Explains the architecture and optimizations
- Includes troubleshooting tips
- Properly credits the original work