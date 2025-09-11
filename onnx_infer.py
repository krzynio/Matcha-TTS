#!/usr/bin/env python3
"""
ONNX inference script for Matcha-TTS models.
Usage: python onnx_infer.py marja.onnx "Your text here"
"""

import argparse
import tempfile
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from matcha.text import text_to_sequence, sequence_to_text
from matcha.utils.utils import intersperse

# ---------- Configuration ----------

# ---------- Text processing logic ----------
def process_text_for_onnx(text, cleaners="polish_cleaners"):
    """Process text similar to inference.py but adapted for ONNX"""
    seq = text_to_sequence(text, [cleaners])[0]
    x = torch.tensor(intersperse(seq, 0), dtype=torch.long)[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    return x.numpy(), x_lengths.numpy(), x_phones

# ---------- ONNX Synthesis logic ---------- 
def synthesize_onnx(onnx_model, text, cleaners="polish_cleaners", temperature=0.667, length_scale=1.0, speaker_id=None):
    """Synthesize speech using ONNX model"""
    # Process text
    x, x_lengths, phones = process_text_for_onnx(text, cleaners)
    
    # Prepare inputs for ONNX model
    inputs = {
        "x": x,
        "x_lengths": x_lengths,
        "scales": np.array([temperature, length_scale], dtype=np.float32),
    }
    
    # Check if model expects speaker ID
    model_inputs = onnx_model.get_inputs()
    is_multi_speaker = len(model_inputs) == 4
    if is_multi_speaker:
        if speaker_id is None:
            speaker_id = 0
            warnings.warn("[!] Speaker ID not provided! Using speaker ID 0", UserWarning)
        inputs["spks"] = np.repeat(speaker_id, x.shape[0]).astype(np.int64)
    
    # Run inference
    print("[🍵] Generating mel using ONNX Matcha model")
    t0 = perf_counter()
    outputs = onnx_model.run(None, inputs)
    mel_infer_secs = perf_counter() - t0
    
    # Assume model has embedded vocoder (since that's what we're using)
    print("Using ONNX model with embedded vocoder")
    waveform, wav_lengths = outputs
    
    # Handle different output formats
    if len(waveform.shape) > 1:
        waveform = waveform[0]  # Get first sample
    if wav_lengths is not None and len(wav_lengths) > 0:
        wav_length = int(wav_lengths[0])
        waveform = waveform[:wav_length]  # Trim to actual length
    
    total_infer_secs = mel_infer_secs
    wav_secs = len(waveform) / 22050
    rtf = total_infer_secs / wav_secs if wav_secs > 0 else 0
    
    print(f"Inference time: {total_infer_secs:.3f}s")
    print(f"Generated audio length: {wav_secs:.3f}s")
    print(f"Real-time factor: {rtf:.3f}")
    
    return phones, waveform

def main():
    parser = argparse.ArgumentParser(description="Generate speech using ONNX Matcha-TTS model")
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--file", help="Text file with sentences (one per line) for batch processing")
    parser.add_argument("--cleaners", default="polish_cleaners", 
                       help="Text cleaners to use (default: polish_cleaners)")
    parser.add_argument("--temperature", type=float, default=0.667,
                       help="Sampling temperature (default: 0.667)")
    parser.add_argument("--length_scale", type=float, default=1.0,
                       help="Length scale (default: 1.0)")
    parser.add_argument("--speaker", type=int, default=None,
                       help="Speaker ID for multi-speaker models")
    parser.add_argument("--output", help="Output WAV file path (for single text)")
    parser.add_argument("--name", help="Name for the output file (will be saved as name.wav)")
    parser.add_argument("--output_dir", help="Output directory for batch processing")
    parser.add_argument("--gpu", action="store_true", 
                       help="Use GPU for ONNX inference (default: CPU)")
    parser.add_argument("--fast", action="store_true",
                       help="Enable fast inference optimizations")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.text and not args.file:
        print("Error: Either --text or --file must be provided")
        return
        
    if not Path(args.model).exists():
        print(f"Error: ONNX model file {args.model} not found")
        return
    
    # Set up ONNX runtime with optimizations
    session_options = ort.SessionOptions()
    
    if args.fast:
        # Fast inference optimizations
        session_options.enable_cpu_mem_arena = True   # Enable for speed
        session_options.enable_mem_pattern = True    # Enable for speed
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # Faster for single inference
        print("Fast mode enabled - optimized for speed over memory")
    else:
        # Balanced optimizations
        session_options.enable_cpu_mem_arena = False  # Reduce memory usage
        session_options.enable_mem_pattern = False    # Reduce memory usage  
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL  # Enable parallel execution
    
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.inter_op_num_threads = 0  # Use all available cores
    session_options.intra_op_num_threads = 0  # Use all available cores
    
    if args.gpu and torch.cuda.is_available():
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
                "cudnn_conv_algo_search": "EXHAUSTIVE",    # Best performance
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider"
        ]
    else:
        providers = ["CPUExecutionProvider"]
    
    try:
        onnx_model = ort.InferenceSession(
            args.model, 
            sess_options=session_options,
            providers=providers
        )
        print(f"Loaded ONNX model: {args.model}")
        print(f"Using providers: {onnx_model.get_providers()}")
        print(f"Optimization level: {session_options.graph_optimization_level}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return
    
    # Process text(s)
    try:
        if args.file:
            # Batch processing
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f.readlines() if line.strip()]
            
            output_dir = Path(args.output_dir) if args.output_dir else Path("outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing {len(texts)} texts in batch mode...")
            total_start = perf_counter()
            
            for i, text in enumerate(texts, 1):
                print(f"\n[{i}/{len(texts)}] Processing: {text[:50]}{'...' if len(text) > 50 else ''}")
                
                phones, waveform = synthesize_onnx(
                    onnx_model, text, args.cleaners,
                    args.temperature, args.length_scale, args.speaker
                )
                
                # Save each audio file
                output_path = output_dir / f"output_{i:03d}.wav"
                sf.write(output_path, waveform, 22050, "PCM_24")
                print(f"Saved: {output_path}")
            
            total_time = perf_counter() - total_start
            avg_time = total_time / len(texts)
            print(f"\nBatch complete! Total time: {total_time:.2f}s, Average: {avg_time:.2f}s per text")
            
        else:
            # Single text processing
            phones, waveform = synthesize_onnx(
                onnx_model, args.text, args.cleaners,
                args.temperature, args.length_scale, args.speaker
            )
            
            # Print phonetic representation
            print(f"Phonetic representation: {phones}")
            
            # Save audio file
            if args.output:
                sf.write(args.output, waveform, 22050, "PCM_24")
                print(f"Audio saved to {args.output}")
            elif args.name:
                output_path = f"{args.name}.wav"
                sf.write(output_path, waveform, 22050, "PCM_24")
                print(f"Audio saved to {output_path}")
            else:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                    sf.write(fp.name, waveform, 22050, "PCM_24")
                    print(f"Audio saved to temporary file: {fp.name}")
                
    except Exception as e:
        print(f"Error during synthesis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()