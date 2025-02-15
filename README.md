# ivrit-ai-whisper-claude-transcriber
A sophisticated audio/video transcription system designed for Hebrew language processing, leveraging the ivrit-ai/faster-whisper-v2-d4 model with Claude AI text enhancement.

**Original model**: [ivrit-ai/faster-whisper-v2-d4](https://huggingface.co/ivrit-ai/faster-whisper-v2-d4)  
**Implementation**: [HaKochav](https://github.com/HaKochav), 2025

## System Architecture

The system operates through several coordinated components:

1. **Audio Processing**:
   - Automated file format conversion via FFmpeg
   - VAD (Voice Activity Detection) filtering
   - Segmentation with 2-second overlap

2. **Transcription Engine**:
   - Whisper model optimization
   - Batch processing with memory management
   - Error recovery and state persistence

3. **Text Enhancement** (Optional):
   - Claude AI integration
   - Chunk-based processing
   - Format preservation

## Sample Output

The system includes a sample transcription of Israel's Declaration of Independence ([View Original Video](https://www.youtube.com/watch?v=VlOGvqSSekc)), demonstrating all output formats and system capabilities. A detailed analysis and explanation of this sample can be found in the `sample_output/README.md` file.

## System Requirements

### Software Requirements and Verified Versions

The system requires the following components. For each component, we specify the version that has been thoroughly tested and verified:

1. **Core Requirements**:
   - Python (Tested with version 3.11.9)
   - CUDA for GPU support (Tested with 12.6)
   - FFmpeg (Tested with 2025-02-06-git-6da82b4485)

2. **Python Packages**:
   The following package versions have been verified to work together:
   ```bash
   # Core ML and Audio Processing
   torch==2.6.0+cu126
   torchvision==0.21.0+cu126
   torchaudio==2.6.0+cu126
   faster-whisper==1.1.1

   # Support Libraries
   tqdm==4.67.1
   typing-extensions==4.12.2
   
   # Optional Text Enhancement
   anthropic==0.45.2  # Required only for Claude integration
   ```

Note: While newer versions might work, the above configurations have been thoroughly tested and verified for stability. If using different versions, please test thoroughly before production use.

### Hardware Support
The system supports both GPU and CPU operation:

1. **GPU Mode** (Recommended):
   - NVIDIA GPU with CUDA support
   - Automatically utilizes float16 computation

2. **CPU Mode**:
   - Automatically selected if GPU unavailable
   - Uses int8 computation
   - Multi-threaded processing

## Installation Process

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **FFmpeg Installation**:
   - Install FFmpeg version 2025-02-06-git-6da82b4485
   - Add to system PATH
   - Verify: `ffmpeg -version`

## Usage Guide

1. **Basic Operation**:
   ```bash
   python transcribe.py
   ```

2. **Interactive Setup**:
   The script will prompt for:
   - Input directory path (Enter for current directory)
   - Claude API key (optional)
   - Prompt customization preferences

3. **Supported Formats**:
   - Native: MP3, WAV, M4A, MP4
   - Others: Automatic conversion to WAV

## Technical Implementation Details

### Claude AI Integration

The system implements a specific prompt structure for Claude AI text enhancement:
```python
Format your response with:
<improved_text>
```
Note: The closing tag is intentionally omitted in the prompt. This implementation detail appears to improve Claude's response consistency and output format reliability. The system's regex pattern handles both closed and unclosed tag scenarios effectively.

### Real-Time Progress Monitoring

The system maintains a JSON-based progress tracking file that updates in real-time during transcription. To monitor progress:

1. **Recommended Tools**:
   - Visual Studio Code (with built-in JSON preview)
   - Any text editor with live JSON reload capabilities
   - Dedicated JSON viewers with auto-refresh functionality

2. **Monitoring Process**:
   - Open the progress tracking JSON file in your chosen tool
   - Enable auto-refresh if necessary
   - Monitor real-time updates including:
     - Current file progress
     - Processing status
     - Time estimates
     - Error counts

## Configuration Details

The following parameters have been carefully optimized:

1. **Transcription Parameters**:
   ```python
   batch_size = 16
   vad_filter = True
   min_silence_duration_ms = 500
   compute_type = 'float16'  # GPU mode
   compute_type = 'int8'    # CPU mode
   ```

2. **Text Processing**:
   ```python
   CHUNK_MIN_SIZE = 125     # Minimum chunk size for processing
   CHUNK_MAX_SIZE = 2000    # Maximum chunk size for processing
   OVERLAP_SIZE = 100       # Overlap between chunks
   MAX_RETRIES = 3         # Maximum retry attempts
   RETRY_DELAY = 30        # Delay between retries (seconds)
   ```

## Output Structure

The system generates several files:

1. **Transcription Files**:
   - `{filename}_raw.txt`: Original transcription with timestamps
   - `{filename}_processed.txt`: Enhanced text (if using Claude)
   - `{filename}_report.txt`: Comprehensive report, original transcript on top and enhanced text on the bottom.

2. **System Files**:
   - `logs/`: Detailed operation logs
   - `temp/`: Temporary processing files
   - `checkpoints/`: State persistence data

## Advanced Features

1. **Error Recovery**:
   - Automatic retry on transcription failures
   - CUDA error handling
   - Checkpoint-based state recovery

2. **Resource Management**:
   - Automatic GPU memory cleanup
   - Multi-threading optimization
   - Progress persistence

## Error Handling

The system implements comprehensive error handling:

1. **File Processing**:
   - Invalid format detection
   - Conversion error recovery
   - Incomplete file handling

2. **Transcription**:
   - Segment validation
   - Gap detection
   - Resource exhaustion handling

## System Output Files

The system generates three different output files for each transcription, each serving a distinct purpose:

### 1. Raw Transcription (`_raw.txt`)
Contains the direct output from the Whisper AI model:
- Precise timestamps for each speech segment [MM:SS -> MM:SS]
- Raw text exactly as transcribed
- No formatting or content organization
- Useful for:
  - Time-aligned references
  - Subtitle creation
  - Word-for-word transcription needs

### 2. Processed Content (`_processed.txt`)
Enhanced version produced by Claude AI:
- Organized into logical sections with headers
- Bullet points for clear structure
- Improved formatting and readability
- Useful for:
  - Content summarization
  - Document creation
  - Research and analysis

### 3. Complete Report (`_report.txt`)
Comprehensive file containing:
- File metadata and processing information
- Complete raw transcription with timestamps
- Enhanced processed content
- Useful for:
  - Full documentation
  - Quality verification
  - Process auditing

Note: This is the explanation for the output files according to the implemented default prompt. With any other prompt the components of Claude's answer may vary. The raw transcription will stay the same nevertheless.

## Disclaimer and Third-Party Dependencies

This repository contains code that integrates third-party models and tools, including but not limited to **Whisper**, **Claude**, and **Ivrit-AI**. The tools and models used in this repository are not created or owned by the author of this repository. By using this code, you agree to comply with the respective licenses and terms of use of these third-party tools.

- Whisper is governed by its own license (MIT License).
- Claude and Ivrit-AI models have their respective terms of service which users should review and comply with.

The author of this repository is not responsible for any legal issues or consequences arising from the use of these third-party models and tools. Please ensure you review and comply with any applicable licenses and terms of service related to these models.

If you are unsure about the licensing of any particular component, refer to the official repositories or documentation of the respective models.

## License

The code in this repository is licensed under the MIT License. See the LICENSE file for details.

