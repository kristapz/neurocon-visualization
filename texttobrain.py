#!/usr/bin/env python3
"""
text_to_brain_viz.py - Complete pipeline from text to brain visualization
"""

import os
import sys
import json
import torch
import torch.nn as nn
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layers import ClipModel, ProjectionHead, ResidualHead
from src.embeddings import embed_texts
from nilearn.datasets import fetch_atlas_difumo, load_mni152_template
from nilearn.maskers import NiftiMapsMasker
from nilearn.image import get_data, new_img_like


class TextToBrainViz:
    def __init__(self):
        print("Loading model and data...")

        # Load embeddings
        with open('data/preprocessed_train_text_embeddings.pkl', 'rb') as f:
            self.train_text_embeddings = pickle.load(f)
        with open('data/preprocessed_train_gaussian_embeddings.pkl', 'rb') as f:
            self.train_gaussian_embeddings = pickle.load(f)

        print(f"Loaded {len(self.train_text_embeddings)} samples")
        print(f"Text dim: {self.train_text_embeddings.shape[1]}, DiFuMo dim: {self.train_gaussian_embeddings.shape[1]}")

        # Load model
        self.model = self._load_model()

        # Pre-compute latents
        print("Computing latent representations...")
        with torch.no_grad():
            self.train_brain_latents = self.model.encode_image(
                torch.from_numpy(self.train_gaussian_embeddings).float()
            )

        # Setup DiFuMo for reconstruction
        self.dim = self.train_gaussian_embeddings.shape[1]
        print(f"Setting up DiFuMo atlas (dim={self.dim})...")
        self.difumo = fetch_atlas_difumo(dimension=self.dim, resolution_mm=2)
        self.masker = NiftiMapsMasker(self.difumo.maps, standardize=False).fit()

        # Load text encoder
        self._load_text_encoder()

        # Setup output directories
        os.makedirs("output/maps", exist_ok=True)

        # Save MNI underlay once
        if not os.path.exists("mni152_t1_2mm.nii.gz"):
            print("Saving MNI152 underlay...")
            t1 = load_mni152_template(resolution=2)
            nib.save(t1, "mni152_t1_2mm.nii.gz")

        print("Ready!\n")

    def _load_model(self):
        text_dim = self.train_text_embeddings.shape[1]
        output_dim = self.train_gaussian_embeddings.shape[1]

        model = ClipModel(
            image_model=nn.Sequential(
                ResidualHead(output_dim, dropout=0.6),
                ResidualHead(output_dim, dropout=0.6),
                ResidualHead(output_dim, dropout=0.6),
            ),
            text_model=nn.Sequential(
                ProjectionHead(text_dim, output_dim, dropout=0.6),
                ResidualHead(output_dim, dropout=0.6),
                ResidualHead(output_dim, dropout=0.6),
            ),
            logit_scale=10,
            logit_bias=None
        )

        model.load_state_dict(torch.load('best_val.pt', map_location='cpu'))
        model.eval()
        return model

    def _load_text_encoder(self):
        from transformers import AutoTokenizer, AutoModel

        model_name = "EleutherAI/gpt-neo-1.3B"
        print(f"Loading {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm_model = AutoModel.from_pretrained(model_name)
        self.lm_model.eval()

    def text_to_embedding_4096(self, text):
        """Convert text to 4096-dim embedding"""
        base_embedding = embed_texts(
            [text],
            self.tokenizer,
            self.lm_model,
            device=torch.device('cpu')
        )

        if base_embedding.shape[1] == 2048:
            # Duplicate to match training
            embedding_4096 = np.concatenate([base_embedding, base_embedding], axis=1)
        elif base_embedding.shape[1] == 4096:
            embedding_4096 = base_embedding
        else:
            raise ValueError(f"Unexpected embedding size: {base_embedding.shape[1]}")

        return embedding_4096

    def top_percent_threshold(self, img, top_pct=0.15):
        """Keep only the top fraction of positive voxels"""
        data = get_data(img)
        pos = data[data > 0]
        if pos.size == 0:
            return img
        thr = float(np.quantile(pos, 1.0 - top_pct))
        out = np.where(data >= thr, data, 0.0).astype(np.float32)
        return new_img_like(img, out, copy_header=True)

    def text_to_brain_map(self, text, top_pct=0.15):
        """Complete pipeline: text → brain map"""

        print(f"Processing: '{text[:100]}{'...' if len(text) > 100 else ''}'")

        # Get embedding
        text_embedding = self.text_to_embedding_4096(text)

        # Project to latent space
        with torch.no_grad():
            text_latent = self.model.encode_text(
                torch.from_numpy(text_embedding).float()
            )

        # Find most similar brain pattern
        similarities = torch.cosine_similarity(
            text_latent,
            self.train_brain_latents,
            dim=1
        )

        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()

        print(f"Best match: index {best_idx}, similarity {best_score:.4f}")

        # Get DiFuMo coefficients
        difumo_coeffs = self.train_gaussian_embeddings[best_idx]

        # Reconstruct brain volume
        brain_img = self.masker.inverse_transform(difumo_coeffs.reshape(1, -1))

        # Apply threshold for cleaner visualization
        brain_img_sparse = self.top_percent_threshold(brain_img, top_pct=top_pct)

        return brain_img_sparse, best_score, best_idx


def main():
    # Initialize
    converter = TextToBrainViz()

    print("=" * 60)
    print("Text to Brain Visualization")
    print("=" * 60)
    print("Enter text to generate brain activation maps.")
    print("Type 'quit' to exit.\n")

    generated_maps = []

    while True:
        # Get input
        text = input("\nEnter text: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if not text:
            print("Please enter some text.")
            continue

        try:
            # Generate brain map
            brain_img, similarity, idx = converter.text_to_brain_map(text)

            # Save map
            filename = f"output/maps/brain_{len(generated_maps):03d}.nii.gz"
            nib.save(brain_img, filename)
            print(f"Saved: {filename}")

            generated_maps.append({
                'file': filename,
                'text': text[:100],
                'similarity': float(similarity),
                'index': int(idx)
            })

            # Update index.json for viewer
            with open("index.json", "w") as f:
                json.dump({
                    'files': [m['file'] for m in generated_maps],
                    'metadata': generated_maps
                }, f, indent=2)

            print(f"\nVisualization ready!")
            print(f"Maps generated: {len(generated_maps)}")

            if len(generated_maps) == 1:
                print("\nTo view:")
                print("1. Run: python -m http.server 8000")
                print("2. Open: http://localhost:8000/niivue_demo.html")

        except Exception as e:
            print(f"Error: {e}")

    print(f"\nGenerated {len(generated_maps)} brain maps.")
    print("Goodbye!")


if __name__ == "__main__":
    main()