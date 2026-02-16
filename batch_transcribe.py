import os
import whisper
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch transcribe audio files in the 'audio' folder.")
    parser.add_argument("--language", type=str, help="Language for transcription (e.g., 'en', 'sv'). If not provided, Whisper will auto-detect.")
    parser.add_argument("--model", type=str, default="turbo", help="Whisper model to use (default: turbo).")
    args = parser.parse_args()

    audio_dir = Path("audio")
    text_dir = Path("text")

    # Ensure output directory exists
    text_dir.mkdir(exist_ok=True)

    # Load the model
    print(f"Loading Whisper model: {args.model}...")
    model = whisper.load_model(args.model)

    # Supported audio extensions
    extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

    audio_files = [f for f in audio_dir.iterdir() if f.suffix.lower() in extensions]

    if not audio_files:
        print("No audio files found in the 'audio' folder.")
        return

    print(f"Found {len(audio_files)} audio files.")

    success_count = 0
    failure_count = 0

    for audio_file in audio_files:
        output_file = text_dir / f"{audio_file.stem}.txt"
        
        if output_file.exists():
            print(f"Skipping {audio_file.name}, transcription already exists.")
            continue

        print(f"Transcribing {audio_file.name}...")
        
        # Transcribe
        transcribe_options = {}
        if args.language:
            transcribe_options["language"] = args.language

        try:
            result = model.transcribe(str(audio_file), **transcribe_options)
            
            # Save the text
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result["text"].strip())
            
            print(f"Saved transcription to {output_file}")
            success_count += 1
        except Exception as e:
            print(f"Error transcribing {audio_file.name}: {e}")
            failure_count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully transcribed: {success_count}")
    print(f"Failed: {failure_count}")

if __name__ == "__main__":
    main()
