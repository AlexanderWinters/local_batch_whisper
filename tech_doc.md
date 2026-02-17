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
3.  **GPU Detection & Parallelization**:
    - The script automatically detects available CUDA GPUs.
    - On macOS, it checks for MPS (Metal Performance Shaders) support.
    - If multiple GPUs are found, it spawns multiple parallel workers, one for each GPU.
    - If no GPU is found, it defaults to CPU.
4.  **Model Loading**: Loads the specified Whisper model into each worker process on the assigned device.
5.  **File Iteration**: 
    - Scans the `audio/` folder for files with common audio extensions (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`).
    - Skips files if a corresponding `.txt` file already exists in the `text/` folder.
6.  **Transcription**:
    - Uses `ProcessPoolExecutor` to run transcriptions in parallel.
    - Calls `model.transcribe()` for each file within its own process.
7.  **Output**:
    - Saves the transcribed text into a `.txt` file in the `text/` folder.
    - The output file name matches the audio file's base name.

## Usage

Run the script from the terminal:

```bash
python batch_transcribe.py --language sv
```

### Command-line Arguments

- `--language`: Optional. Specifies the language of the audio files (e.g., `sv` for Swedish). If omitted, Whisper attempts to auto-detect the language.
- `--model`: Optional. Specifies which Whisper model to use. Defaults to `turbo`.
- `--workers`: Optional. Number of parallel workers to spawn. Defaults to the number of detected GPUs.
- `--device_ids`: Optional. Comma-separated list of device IDs to use (e.g., `0,1` for CUDA GPUs 0 and 1, or `mps` for macOS).

### Examples

Use specific GPUs:
```bash
python batch_transcribe.py --device_ids 0,1
```

Manually set number of workers (e.g., to run 4 transcriptions on 2 GPUs):
```bash
python batch_transcribe.py --workers 4 --device_ids 0,1
```

To see all options:

```bash
python batch_transcribe.py --help
```

## Error Handling

The script is designed to be robust. It includes a `try-except` block around the transcription and file-saving process for each individual audio file. This ensures that if an error occurs (e.g., corrupted audio file, I/O error), the script logs the error and automatically continues to the next file in the queue rather than stopping the entire batch process.

At the end of the execution, the script provides a summary of:
- Total files processed.
- Number of successful transcriptions.
- Number of failed transcriptions.


## Troubleshooting

### Certificates error

I came across the following error:

```
File "/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/urllib/request.py", line 1322, in do_open
raise URLError(err)
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1028)>
```

I installed `certifi` and passed it's paramaters to an env variable.

```
python -m pip install certifi

export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
python batch_transcribe.py
```