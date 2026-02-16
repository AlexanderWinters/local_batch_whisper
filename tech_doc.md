# Technical Documentation: Batch Audio Transcription Script

This document provides technical details about the `batch_transcribe.py` script.

## Overview

The `batch_transcribe.py` script is designed to automate the transcription of multiple audio files stored in a specific directory using OpenAI's Whisper model. It supports language selection and handles basic errors and existing transcriptions.

## Requirements

- Python 3.x
- `openai-whisper` library
- `argparse`, `pathlib` (standard libraries)
- `ffmpeg` (required by Whisper for audio processing)

## Script Logic

1.  **Argument Parsing**: The script uses `argparse` to handle command-line arguments:
    - `--language`: Optional. Specifies the language of the audio files (e.g., `sv` for Swedish). If omitted, Whisper attempts to auto-detect the language.
    - `--model`: Optional. Specifies which Whisper model to use. Defaults to `turbo`.
2.  **Directory Setup**:
    - It looks for audio files in the `audio/` directory.
    - It ensures the `text/` directory exists for output.
3.  **Model Loading**: Loads the specified Whisper model into memory.
4.  **File Iteration**: 
    - Scans the `audio/` folder for files with common audio extensions (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`).
    - Skips files if a corresponding `.txt` file already exists in the `text/` folder.
5.  **Transcription**:
    - Calls `model.transcribe()` for each file.
    - Passes the `language` parameter if provided.
6.  **Output**:
    - Saves the transcribed text into a `.txt` file in the `text/` folder.
    - The output file name matches the audio file's base name.

## Usage

Run the script from the terminal:

```bash
python batch_transcribe.py --language sv
```

To see all options:

```bash
python batch_transcribe.py --help
```

## Error Handling

The script includes a `try-except` block around the transcription process to catch and report errors (e.g., corrupted files, memory issues) without stopping the entire batch process.
