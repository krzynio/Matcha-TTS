#!/usr/bin/env python3
"""
Command-line client for Gosia model.
Usage: python gosia_cli.py "Your text here"
"""

import argparse
import tempfile
import soundfile as sf
import torch
from pathlib import Path
from time import perf_counter

from matcha.cli import (
    get_device,
    load_vocoder,
    to_waveform,
    assert_model_downloaded,
    VOCODER_URLS,
)
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import text_to_sequence, sequence_to_text
from matcha.utils.utils import intersperse, plot_tensor, get_user_data_dir
from matcha.hifigan.denoiser import Denoiser
from huggingface_hub import hf_hub_download

# ---------- Configuration ----------
VOICE_OPTIONS = {
    "Polish – Gosia": {
        "repo": "jimregan/matcha-pl-gosia",
        "ckpt": "checkpoints/last.ckpt",
        "cleaners": "polish_cleaners"
    },
    "Polish – Darkman": {
        "repo": "jimregan/matcha-pl-darkman",
        "ckpt": "checkpoints/last.ckpt",
        "cleaners": "polish_cleaners"
    },
    "Hungarian – Anna": {
        "repo": "jimregan/matcha-hu-anna",
        "ckpt": "checkpoints/last.ckpt",
        "cleaners": "hungarian_cleaners"
    },
}

VOCODER_NAME = "hifigan_univ_v1"
LOCATION = Path(get_user_data_dir())

# Fix device initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Download vocoder with built-in logic ----------
vocoder_path = LOCATION / VOCODER_NAME
assert_model_downloaded(vocoder_path, VOCODER_URLS[VOCODER_NAME])
vocoder, denoiser = load_vocoder(VOCODER_NAME, vocoder_path, device)

# ---------- Download voices from Hugging Face ----------
models = {}
for label, info in VOICE_OPTIONS.items():
    ckpt_path = hf_hub_download(repo_id=info["repo"], filename=info["ckpt"])
    model = MatchaTTS.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    models[label] = {**info, "model": model}

# ---------- Synthesis logic ----------
@torch.inference_mode()
def process_text(text, cleaners):
    seq = text_to_sequence(text, [cleaners])[0]
    x = torch.tensor(intersperse(seq, 0), dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return x, x_lengths, x_phones

@torch.inference_mode()
def synthesize(voice, text, n_timesteps=10, temperature=0.667, length_scale=1.0):
    info = voice
    x, x_lengths, phones = process_text(text, info["cleaners"])
    
    # Time the model synthesis
    print("[🍵] Generating mel using Matcha model")
    mel_t0 = perf_counter()
    output = info["model"].synthesise(
        x, x_lengths,
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=None,
        length_scale=length_scale
    )
    mel_infer_secs = perf_counter() - mel_t0
    
    # Time the vocoder
    print("Generating waveform from mel using HiFiGAN vocoder")
    vocoder_t0 = perf_counter()
    waveform = to_waveform(output["mel"], vocoder, denoiser)
    vocoder_infer_secs = perf_counter() - vocoder_t0
    
    # Calculate timing statistics
    total_infer_secs = mel_infer_secs + vocoder_infer_secs
    wav_secs = len(waveform) / 22050
    rtf = total_infer_secs / wav_secs if wav_secs > 0 else 0
    
    print(f"Mel inference time: {mel_infer_secs:.3f}s")
    print(f"Vocoder inference time: {vocoder_infer_secs:.3f}s")
    print(f"Total inference time: {total_infer_secs:.3f}s")
    print(f"Generated audio length: {wav_secs:.3f}s")
    print(f"Real-time factor: {rtf:.3f}")
    
    return phones, waveform

def main():
    parser = argparse.ArgumentParser(description="Generate speech using the Gosia model")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("--voice", choices=list(VOICE_OPTIONS.keys()), 
                       default="Polish – Gosia", help="Voice to use")
    parser.add_argument("--n_timesteps", type=int, default=10, 
                       help="Number of ODE steps")
    parser.add_argument("--temperature", type=float, default=0.667,
                       help="Sampling temperature")
    parser.add_argument("--length_scale", type=float, default=1.0,
                       help="Length scale")
    parser.add_argument("--output", help="Output WAV file path")
    parser.add_argument("--name", help="Name for the output file (will be saved as name.wav)")
    parser.add_argument("--cleaners", default="polish_cleaners", 
                       help="Text cleaners to use (default: polish_cleaners)")
    parser.add_argument("--checkpoint", help="Path to custom checkpoint file")
    
    args = parser.parse_args()
    
    # Use custom checkpoint if provided, otherwise use voice from VOICE_OPTIONS
    if args.checkpoint:
        # Load custom checkpoint
        if not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint file {args.checkpoint} not found")
            return
        
        # Create a temporary model entry for custom checkpoint
        custom_model = MatchaTTS.load_from_checkpoint(args.checkpoint, map_location=device)
        custom_model.eval()
        custom_voice_info = {
            "model": custom_model,
            "cleaners": args.cleaners
        }
        voice_to_use = custom_voice_info
        voice_label = "Custom Checkpoint"
    else:
        # Use predefined voice
        voice_to_use = models[args.voice]
        voice_label = args.voice
    
    # Synthesize the text
    phones, waveform = synthesize(
        voice_to_use, 
        args.text,
        args.n_timesteps,
        args.temperature,
        args.length_scale
    )
    
    # Print phonetic representation
    print(f"Phonetic representation: {phones}")
    print(f"Using voice: {voice_label}")
    
    # Save audio file if output path is provided
    if args.output:
        sf.write(args.output, waveform, 22050, "PCM_24")
        print(f"Audio saved to {args.output}")
    elif args.name:
        # Save to file with specified name
        output_path = f"{args.name}.wav"
        sf.write(output_path, waveform, 22050, "PCM_24")
        print(f"Audio saved to {output_path}")
    else:
        # Save to temporary file and print path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            sf.write(fp.name, waveform, 22050, "PCM_24")
            print(f"Audio saved to temporary file: {fp.name}")

if __name__ == "__main__":
    main()
