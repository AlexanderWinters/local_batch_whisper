import os
import whisper
import argparse
from pathlib import Path
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def transcribe_file(audio_path, model_name, language, device):
    """Worker function to transcribe a single file on a specific device."""
    try:
        # Load model inside the process to ensure it's on the correct GPU
        model = whisper.load_model(model_name, device=device)
        
        output_dir = Path("text")
        output_file = output_dir / f"{audio_path.stem}.txt"
        
        print(f"[{device}] Transcribing {audio_path.name}...")
        
        transcribe_options = {}
        if language:
            transcribe_options["language"] = language

        result = model.transcribe(str(audio_path), **transcribe_options)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result["text"].strip())
        
        print(f"[{device}] Saved transcription to {output_file}")
        return True
    except Exception as e:
        print(f"Error transcribing {audio_path.name} on {device}: {e}")
        return False

def get_available_devices():
    """Detect available GPUs (CUDA). If none, fallback to CPU. On macOS, consider MPS."""
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif torch.backends.mps.is_available():
        # MPS doesn't support multiple 'devices' in the same way CUDA does usually, 
        # but we can still return it as the primary device.
        return ["mps"]
    else:
        return ["cpu"]

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files in the 'audio' folder.")
    parser.add_argument("--language", type=str, help="Language for transcription (e.g., 'en', 'sv'). If not provided, Whisper will auto-detect.")
    parser.add_argument("--model", type=str, default="turbo", help="Whisper model to use (default: turbo).")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers. Defaults to number of available GPUs.")
    parser.add_argument("--device_ids", type=str, help="Comma-separated list of device IDs to use (e.g., '0,1' for CUDA or 'mps').")
    args = parser.parse_args()

    audio_dir = Path("audio")
    text_dir = Path("text")

    # Ensure output directory exists
    text_dir.mkdir(exist_ok=True)

    # Detect devices
    if args.device_ids:
        devices = [d.strip() for d in args.device_ids.split(",")]
        # If user provides just numbers, assume cuda
        devices = [f"cuda:{d}" if d.isdigit() else d for d in devices]
    else:
        devices = get_available_devices()

    # Determine number of workers
    num_workers = args.workers if args.workers is not None else len(devices)
    
    # If we have more workers than devices, we cycle through devices
    worker_devices = [devices[i % len(devices)] for i in range(num_workers)]

    print(f"Using devices: {', '.join(set(worker_devices))}")
    print(f"Number of parallel workers: {num_workers}")

    # Supported audio extensions
    extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    audio_files = [f for f in audio_dir.iterdir() if f.suffix.lower() in extensions]

    # Filter out already transcribed files
    audio_files_to_process = []
    for f in audio_files:
        if not (text_dir / f"{f.stem}.txt").exists():
            audio_files_to_process.append(f)
        else:
            print(f"Skipping {f.name}, transcription already exists.")

    if not audio_files_to_process:
        if not audio_files:
            print("No audio files found in the 'audio' folder.")
        else:
            print("All audio files already have transcriptions.")
        return

    print(f"Found {len(audio_files_to_process)} audio files to process.")

    success_count = 0
    failure_count = 0

    # Use ProcessPoolExecutor for parallel processing
    # Note: We need to use 'spawn' start method for CUDA to avoid issues
    if torch.cuda.is_available():
        multiprocessing.set_start_method('spawn', force=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, audio_file in enumerate(audio_files_to_process):
            device = worker_devices[i % num_workers]
            futures.append(executor.submit(transcribe_file, audio_file, args.model, args.language, device))
        
        for future in futures:
            if future.result():
                success_count += 1
            else:
                failure_count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully transcribed: {success_count}")
    print(f"Failed: {failure_count}")

if __name__ == "__main__":
    main()
