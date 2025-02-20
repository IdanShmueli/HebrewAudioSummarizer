import faster_whisper
import os
import sys
import json
import time
import wave
import torch
import ffmpeg
import psutil
import pathlib
import asyncio
import logging
import threading
import contextlib
import subprocess
import anthropic
import gc
import signal
import re
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from functools import wraps
from tqdm import tqdm

# Configure rich console 
console = Console()

# Configure logging with rich formatting and file output
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8')  
    ]
)

@dataclass
class TranscriptionProgress:
    """Tracks the progress of a single file's transcription process"""
    filename: str
    total_duration: float = 0.0
    current_position: float = 0.0 
    current_action: str = "Initializing"
    processed_duration: float = 0.0
    start_time: datetime = None
    estimated_completion: datetime = None
    current_chunk: int = 0
    total_chunks: int = 0
    status: str = "pending"
    error_count: int = 0
    retries: int = 0
    
    def to_dict(self) -> dict:
        """Convert progress to dictionary for JSON serialization"""
        return {
            'filename': self.filename,
            'total_duration': self.total_duration,
            'current_position': self.current_position,
            'current_action': self.current_action,
            'processed_duration': self.processed_duration, 
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'current_chunk': self.current_chunk,
            'total_chunks': self.total_chunks,
            'status': self.status,
            'error_count': self.error_count,
            'retries': self.retries
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TranscriptionProgress':
        """Create progress object from dictionary"""
        if 'start_time' in data and data['start_time']:
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if 'estimated_completion' in data and data['estimated_completion']:
            data['estimated_completion'] = datetime.fromisoformat(data['estimated_completion'])
        return cls(**data)
    
    def update_eta(self):
        """Update estimated completion time based on current progress"""
        if self.total_duration > 0 and self.processed_duration > 0:
            if not self.start_time:
                self.start_time = datetime.now()
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed > 0:
                rate = self.processed_duration / elapsed
                remaining_duration = self.total_duration - self.processed_duration
            
                if rate > 0:
                    remaining_seconds = remaining_duration / rate
                    self.estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)

class ProgressTracker:
    """Manages progress tracking and reporting for the transcription process"""
    def __init__(self):
        self.progress_file = Path("transcription_progress.json")
        self.active_progresses: Dict[str, TranscriptionProgress] = {}
        self.completed_files: Set[str] = set()
        self.load_progress()
        
        # Initialize rich progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        )
        
        # Create progress bars for overall progress
        self.overall_progress = None
        self.file_progress = None 
        self.chunk_progress = None
    
    def load_progress(self):
        """Load progress from JSON file"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.active_progresses = {
                        k: TranscriptionProgress.from_dict(v) 
                        for k, v in data['active_progresses'].items()
                    }
                    self.completed_files = set(data['completed_files'])
                logging.info(f"Loaded progress: {len(self.completed_files)} completed files")
        except Exception as e:
            logging.error(f"Error loading progress: {str(e)}")
            self.active_progresses = {}
            self.completed_files = set()
    
    def save_progress(self):
        """Save progress to JSON file"""
        try:
            data = {
                'active_progresses': {
                    k: v.to_dict() for k, v in self.active_progresses.items()
                },
                'completed_files': list(self.completed_files),
                'last_update': datetime.now().isoformat()
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving progress: {str(e)}")
    
    def get_file_duration(self, file_path: Path) -> float:
        """Get duration of audio/video file using ffmpeg"""
        try:
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            logging.error(f"Error getting duration for {file_path}: {str(e)}")
            return 0.0
    
    def initialize_progress(self, filename: str) -> TranscriptionProgress:
        """Initialize progress tracking for a file"""
        progress = TranscriptionProgress(filename=filename)
        self.active_progresses[filename] = progress
        return progress
    
    def update_progress(self, filename: str, **kwargs):
        """Update progress for a specific file"""
        if filename in self.active_progresses:
            progress = self.active_progresses[filename]
            for key, value in kwargs.items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
            progress.update_eta()
            self.save_progress()
    
    def mark_completed(self, filename: str):
        """Mark a file as completed"""
        self.completed_files.add(filename)
        if filename in self.active_progresses:
            del self.active_progresses[filename]
        self.save_progress()
    
    def is_completed(self, filename: str) -> bool:
        """Check if a file has been completed"""
        return filename in self.completed_files
    
    def get_progress(self, filename: str) -> Optional[TranscriptionProgress]:
        """Get progress for a specific file"""
        return self.active_progresses.get(filename)

def get_system_stats() -> Dict[str, Any]:
    """Get current system resource usage"""
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'gpu_memory_used': torch.cuda.memory_allocated() // 1024 // 1024 if torch.cuda.is_available() else 0,
        'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 if torch.cuda.is_available() else 0
    }

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    return str(timedelta(seconds=int(seconds)))

def create_progress_table() -> Table:
    """Create a rich table for displaying progress"""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File")
    table.add_column("Progress")
    table.add_column("Status")
    table.add_column("ETA")
    return table

class EnhancedWhisperModel:
    """Enhanced Whisper model wrapper with progress tracking and error handling"""
    
    def __init__(self, model_name: str, progress_tracker: ProgressTracker):
        """
        Initialize the model wrapper.
        
        Args:
            model_name (str): Name of the Whisper model to use
            progress_tracker (ProgressTracker): Progress tracking instance
        """
        self.model_name = model_name
        self.progress_tracker = progress_tracker
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the Whisper model with error handling"""
        try:
            logging.info(f"Initializing Whisper model '{self.model_name}' on {self.device}")
            
            # Clear any existing CUDA memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.model = faster_whisper.WhisperModel(
                self.model_name,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8",
                cpu_threads=4,
                num_workers=1
            )
            
            # Verify model initialization
            if not self.model:
                raise RuntimeError("Model initialization failed")
                
            logging.info("Model initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    def clear_gpu_memory(self):
        """Clear GPU memory safely with verification"""
        if self.device == "cuda":
            try:
                # Synchronize before cleanup
                torch.cuda.synchronize()
                
                # Clear memory caches
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Verify cleanup
                allocated = torch.cuda.memory_allocated()
                if allocated > 0:
                    logging.warning(f"GPU memory still allocated: {allocated/1024**2:.2f}MB")
                    
                # Final synchronization
                torch.cuda.synchronize()
                
            except Exception as e:
                logging.error(f"Error clearing GPU memory: {str(e)}")

    def _transcribe_sync(self, file_path: str, progress: TranscriptionProgress) -> Optional[Tuple[Any, Any]]:
        """
        Synchronous transcription method to be run in a separate thread.
        
        Args:
            file_path (str): Path to the audio file
            progress (TranscriptionProgress): Progress tracking object
        
        Returns:
            Optional[Tuple[Any, Any]]: Tuple of segments iterator and info, or None if error
        """
        try:
            # Perform transcription with Whisper model
            segments_with_info = self.model.transcribe(
                file_path,
                language='he',
                vad_filter=True,
                vad_parameters={
                    'min_silence_duration_ms': 500,
                    'speech_pad_ms': 400,
                    'threshold': 0.35
                },
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=None,
                word_timestamps=False
            )
            
            # Validate transcription result
            if not segments_with_info or len(segments_with_info) != 2:
                raise ValueError("Invalid transcription result format")
            
            segments, info = segments_with_info
            
            # Basic validation of segments
            if not segments:
                logging.warning("No segments detected in audio")
            
            return segments, info
            
        except Exception as e:
            logging.error(f"Error in synchronous transcription: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    async def transcribe_file(self, file_path: Path, progress: TranscriptionProgress) -> Optional[List[Dict[str, Any]]]:
        """
        Transcribe an audio file with enhanced error handling and memory management.
        
        Args:
            file_path (Path): Path to the audio file
            progress (TranscriptionProgress): Progress tracking object
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of transcription segments or None if failed
        """
        try:
            # Validate file
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            if file_path.stat().st_size == 0:
                raise ValueError(f"Empty file: {file_path}")
            
            # Update progress
            progress.current_action = "Analyzing audio file"
            self.progress_tracker.update_progress(
                progress.filename,
                status="processing",
                current_action=progress.current_action
            )
            
            # Clear GPU memory before transcription
            self.clear_gpu_memory()
            
            # Get file duration
            duration = self.progress_tracker.get_file_duration(file_path)
            progress.total_duration = duration
            
            # Run transcription in a separate thread
            transcription_result = await asyncio.to_thread(
                self._transcribe_sync, 
                str(file_path),
                progress
            )

            if transcription_result is None:
                raise RuntimeError("Transcription failed")

            segments, info = transcription_result
            
            # Process and validate segments
            processed_segments = []
            last_end = 0.0  # Track segment continuity
            
            for segment in segments:
                # Validate segment timing
                if segment.start < 0 or segment.end <= segment.start:
                    logging.warning(f"Invalid segment timing: start={segment.start}, end={segment.end}")
                    continue
                    
                # Check for gaps
                if segment.start - last_end > 1.0:  # Gap larger than 1 second
                    logging.warning(f"Gap detected between segments: {last_end:.2f} -> {segment.start:.2f}")
                
                # Update tracking
                last_end = segment.end
                
                # Create segment data
                segment_data = {
                    'start': float(segment.start),
                    'end': float(segment.end),
                    'text': segment.text.strip()
                }
                
                # Validate text
                if not segment_data['text']:
                    logging.warning(f"Empty segment text at {segment_data['start']:.2f}")
                    continue
                
                processed_segments.append(segment_data)
                
                # Update progress
                progress.current_position = segment.end
                progress.processed_duration = segment.end
                progress.current_action = f"Processing segment at {segment.end:.2f}s"
                
                self.progress_tracker.update_progress(
                    progress.filename,
                    current_position=progress.current_position,
                    processed_duration=progress.processed_duration,
                    current_action=progress.current_action
                )
            
            # Clear GPU memory after transcription
            self.clear_gpu_memory()
            
            return processed_segments
            
        except Exception as e:
            logging.error(f"Error transcribing file {file_path}: {str(e)}")
            logging.error(traceback.format_exc())
            progress.error_count += 1
            self.progress_tracker.update_progress(
                progress.filename,
                error_count=progress.error_count,
                status="error"
            )
            return None

class TranscriptionCheckpoint:
    """Manages transcription checkpoints and state persistence"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.current_checkpoint = None
    
    def save_checkpoint(self, filename: str, segments: List[Dict[str, Any]], position: float):
        """Save transcription checkpoint"""
        try:
            checkpoint_data = {
                'filename': filename,
                'last_position': position,
                'segments': segments,
                'timestamp': datetime.now().isoformat()
            }
            
            checkpoint_file = self.checkpoint_dir / f"{filename}.checkpoint.json"
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self.current_checkpoint = checkpoint_file
            logging.info(f"Saved checkpoint for {filename} at position {format_time(position)}")
            
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")
    
    def load_checkpoint(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load transcription checkpoint if exists"""
        try:
            checkpoint_file = self.checkpoint_dir / f"{filename}.checkpoint.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                logging.info(f"Loaded checkpoint for {filename}")
                return checkpoint_data
            return None
            
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}")
            return None
    
    def delete_checkpoint(self, filename: str):
        """Delete transcription checkpoint"""
        try:
            checkpoint_file = self.checkpoint_dir / f"{filename}.checkpoint.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logging.info(f"Deleted checkpoint for {filename}")
                
        except Exception as e:
            logging.error(f"Error deleting checkpoint: {str(e)}")

class FileValidator:
    """Validates and processes audio/video files"""
    
    SUPPORTED_EXTENSIONS = {
        # Audio formats
        '.mp3', '.wav', '.m4a', '.aac', '.wma', '.ogg', '.flac',
        # Video formats
        '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'
    }
    
    @staticmethod
    def is_supported_file(file_path: Path) -> bool:
        """Check if file format is supported"""
        return file_path.suffix.lower() in FileValidator.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def convert_to_wav(input_path: Path, output_dir: Path) -> Optional[Path]:
        """Convert audio/video file to WAV format using ffmpeg"""
        try:
            output_path = output_dir / f"{input_path.stem}.wav"
            
            # Construct ffmpeg command
            command = [
                'ffmpeg',
                '-loglevel', 'error',  # Suppress verbose output
                '-i', str(input_path),
                '-vn',  # Disable video
                '-acodec', 'pcm_s16le',  # Audio codec
                '-ar', '16000',  # Sample rate
                '-ac', '1',  # Mono audio
                '-y',  # Overwrite output
                str(output_path)
            ]
            
            # Run conversion
            subprocess.run(command, check=True)
                
            return output_path
            
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg conversion failed for {input_path}: {str(e)}")
            return None
        
        except Exception as e:
            logging.error(f"Error converting file {input_path}: {str(e)}")
            return None
    
    @staticmethod
    def get_audio_info(file_path: Path) -> Optional[Dict[str, Any]]:
        """Get audio file information using ffmpeg"""
        try:
            probe = ffmpeg.probe(str(file_path))
            
            # Get audio stream
            audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
            if not audio_streams:
                logging.error(f"No audio stream found in {file_path}")
                return None
            
            audio_info = audio_streams[0]
            
            return {
                'duration': float(probe['format']['duration']),
                'sample_rate': int(audio_info.get('sample_rate', 0)),
                'channels': int(audio_info.get('channels', 0)),
                'codec': audio_info.get('codec_name', 'unknown')
            }
            
        except ffmpeg.Error as e:
            logging.error(f"FFmpeg error getting audio info for {file_path}: {str(e)}")
            return None
        
        except Exception as e:
            logging.error(f"Error getting audio info for {file_path}: {str(e)}")
            return None

class TextProcessor:
    """Handles text processing, chunking, and correction using Claude API"""
    
    # Class constants
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    CHUNK_MIN_SIZE: int = 125
    CHUNK_MAX_SIZE: int = 2000
    OVERLAP_SIZE: int = 100
    
    # Claude prompt template
    CLAUDE_PROMPT: str = """As a content specialist, create a detailed and comprehensive summary of this lecture segment in Hebrew. This is part {chunk_number} out of {total_chunks}.

<original_text>
{transcribed_text}
</original_text>

Your task is to transform this into a clear, well-structured summary in Hebrew that:
1. Captures EVERY piece of information from the original text
2. Organizes the content into logical sections with clear headers
3. Uses bullet points to break down complex ideas
4. Maintains all references, names, and numbers  
5. Presents information in a more readable format while preserving all details
6. Uses clear, precise terminology
7. Structures related points under common themes

Guidelines for the summary:
- Don't lose any information - every fact, example, and detail should be included
- Break long explanations into concise, clear bullet points
- Group related information under descriptive headers
- Maintain the exact meaning of concepts
- Keep all numerical references
- Use proper terminology
- Present cause-and-effect relationships clearly

Format your summary using Markdown syntax. Here are some examples of Markdown formatting you should use:
```markdown
# Main Header

## Subheader

- Bullet point 1
- Bullet point 2
  - Sub-bullet point

1. Numbered list item 1
2. Numbered list item 2

**Bold text** for emphasis
```

Create the comprehensive summary in Hebrew using Markdown formatting:
<improved_text>

</improved_text>
"""

    def __init__(self, api_key: str, progress_tracker: ProgressTracker):
        """
        Initialize the TextProcessor with API key and progress tracker.
        
        Args:
            api_key (str): The Claude API key
            progress_tracker (ProgressTracker): Instance of progress tracker
            
        Raises:
            ValueError: If api_key is invalid
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Invalid API key provided")
        self.client = anthropic.Client(api_key=api_key)
        self.progress_tracker = progress_tracker
        # Store processed chunks for reporting
        self.chunk_results = {}

    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into appropriately sized chunks with comprehensive validation.
        
        Args:
            text (str): The input text to be split
            
        Returns:
            List[str]: List of text chunks
        """
        try:
            # Input validation
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")
            
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided")

            chunks = []
            current_pos = 0
            text_length = len(text)

            while current_pos < text_length:
                # Calculate potential chunk end
                chunk_end = min(current_pos + self.CHUNK_MAX_SIZE, text_length)
                
                # If not at text end, look for a proper sentence boundary
                if chunk_end < text_length:
                    # Look ahead for sentence ending punctuation
                    look_ahead = text[chunk_end:min(chunk_end + 100, text_length)]
                    
                    # Check for multiple types of sentence endings
                    for punct in ['.', '!', '?', '।', '۔']:
                        next_punct = look_ahead.find(punct)
                        if next_punct != -1:
                            chunk_end = chunk_end + next_punct + 1
                            break
                    
                    # If no punctuation found, look for other natural breaks
                    if chunk_end == current_pos + self.CHUNK_MAX_SIZE:
                        # Try to break at last space to avoid word splitting
                        last_space = text[current_pos:chunk_end].rfind(' ')
                        if last_space != -1:
                            chunk_end = current_pos + last_space + 1

                # Extract and validate chunk
                chunk = text[current_pos:chunk_end].strip()
                
                # Only add chunks that meet minimum size requirement
                if len(chunk) >= self.CHUNK_MIN_SIZE:
                    chunks.append(chunk)
                    logging.debug(f"Created chunk of length {len(chunk)}")
                else:
                    logging.warning(f"Skipping chunk of length {len(chunk)} < {self.CHUNK_MIN_SIZE}")
                
                # Break if we've reached the end
                if chunk_end >= text_length:
                    break
                
                # Update position with overlap
                current_pos = chunk_end - self.OVERLAP_SIZE
                
                # Prevent infinite loop
                if current_pos >= text_length or current_pos <= 0:
                    break

            # Validate results
            if not chunks:
                logging.error("No valid chunks were created")
                return []
                
            logging.info(f"Successfully split text into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logging.error(f"Error in split_into_chunks: {str(e)}")
            return []

    async def process_chunk(self, chunk: str, chunk_number: int, total_chunks: int, progress: TranscriptionProgress) -> Optional[str]:
        """
        Process a single chunk with comprehensive error handling.
        
        Args:
            chunk (str): Text chunk to process
            chunk_number (int): Current chunk number
            total_chunks (int): Total number of chunks
            progress (TranscriptionProgress): Progress tracking object
            
        Returns:
            Optional[str]: Processed text if successful, None otherwise
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Update progress
                progress.current_chunk = chunk_number
                progress.total_chunks = total_chunks
                progress.current_action = f"Processing chunk {chunk_number}/{total_chunks}"
                self.progress_tracker.update_progress(
                    progress.filename,
                    current_chunk=chunk_number,
                    total_chunks=total_chunks,
                    current_action=progress.current_action
                )
                
                # Validate input
                if not isinstance(chunk, str) or not chunk.strip():
                    raise ValueError("Invalid or empty chunk provided")

                # Prepare request
                prompt = self.CLAUDE_PROMPT.format(
                    chunk_number=chunk_number,
                    total_chunks=total_chunks,
                    transcribed_text=chunk
                )
                
                # Make API request in thread pool
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=8192,
                        messages=[{"role": "user", "content": prompt}]
                    )
                )
                
                # Extract content correctly
                if not hasattr(response, 'content') or not response.content:
                    raise ValueError("Response missing content")
                
                content = response.content[0].text if isinstance(response.content, list) else response.content
                
                if not isinstance(content, str):
                    raise TypeError(f"Unexpected content type: {type(content)}")

                # Extract text between improved_text tags
                match = re.search(r'<improved_text>\s*(.*?)(?=<|$)', content, re.DOTALL | re.IGNORECASE)
                if not match:
                    raise ValueError("Failed to find improved_text tags in response")
                
                improved_text = match.group(1).strip()
                if not improved_text:
                    raise ValueError("Empty improved text extracted")
                
                # Store the raw and processed chunks for report
                if progress.filename not in self.chunk_results:
                    self.chunk_results[progress.filename] = {}
                
                self.chunk_results[progress.filename][chunk_number] = {
                    'raw': chunk,
                    'processed': improved_text
                }
                
                return improved_text
                
            except Exception as e:
                logging.error(f"Error processing chunk {chunk_number} (attempt {attempt}/{self.MAX_RETRIES}): {str(e)}")
                progress.error_count += 1
                self.progress_tracker.update_progress(
                    progress.filename,
                    error_count=progress.error_count
                )
                
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.RETRY_DELAY)
                    continue
                
                return None

    async def process_text(self, text: str, progress: TranscriptionProgress) -> Optional[str]:
        """
        Process complete text with chunking and progress tracking.
        
        Args:
            text (str): The text to process
            progress (TranscriptionProgress): Progress tracking object
            
        Returns:
            Optional[str]: Processed text if successful, None otherwise
        """
        try:
            # Validate text
            if not isinstance(text, str):
                raise TypeError(f"Expected string, got {type(text)}")
            
            text = text.strip()
            if not text:
                raise ValueError("Empty text provided")
            
            # Split into chunks
            chunks = self.split_into_chunks(text)
            if not chunks:
                raise ValueError("No valid chunks created")
            
            # Process chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks, 1):
                result = await self.process_chunk(chunk, i, len(chunks), progress)
                if result:
                    processed_chunks.append(result)
                else:
                    logging.error(f"Failed to process chunk {i}")
            
            if not processed_chunks:
                raise ValueError("No chunks were successfully processed")
            
            # Combine results
            return '\n\n'.join(processed_chunks)
            
        except Exception as e:
            logging.error(f"Error in process_text: {str(e)}")
            return None

class TranscriptionOutput:
    """Manages transcription output files and formats"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def save_raw_transcription(self, filename: str, segments: List[Dict[str, Any]]) -> Optional[Path]:
        """Save raw transcription with timestamps"""
        try:
            output_path = self.output_dir / f"{filename}_raw.txt"
            
            with output_path.open('w', encoding='utf-8') as f:
                for segment in segments:
                    f.write(f"[{format_time(segment['start'])} -> {format_time(segment['end'])}]\n")
                    f.write(f"{segment['text']}\n\n")
            
            logging.info(f"Saved raw transcription to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving raw transcription: {str(e)}")
            return None
    
    def save_processed_output(self, filename: str, content: str) -> Optional[Path]:
        """Save processed and formatted output"""
        try:
            output_path = self.output_dir / f"{filename}_processed.txt"
            
            with output_path.open('w', encoding='utf-8') as f:
                f.write(content)
            
            logging.info(f"Saved processed output to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving processed output: {str(e)}")
            return None
    
    def create_full_report(self, filename: str, segments: List[Dict[str, Any]], 
                           processed_content: str, chunk_results: Dict[int, Dict[str, str]]) -> Optional[Path]:
        """
        Create comprehensive report with both raw and processed content, showing chunks side by side.
        
        Args:
            filename (str): The file name
            segments (List[Dict[str, Any]]): List of transcription segments
            processed_content (str): Final processed content
            chunk_results (Dict[int, Dict[str, str]]): Dictionary of raw and processed chunks by number
            
        Returns:
            Optional[Path]: Path to the report file if successful, None otherwise
        """
        try:
            report_path = self.output_dir / f"{filename}_report.txt"
            
            with report_path.open('w', encoding='utf-8') as f:
                # Write header
                f.write("=" * 80 + "\n")
                f.write(f"Transcription Report for: {filename}\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                # Write chunks side-by-side
                if chunk_results:
                    f.write("CHUNKS COMPARISON\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for chunk_num in sorted(chunk_results.keys()):
                        chunk = chunk_results[chunk_num]
                        f.write(f"===== CHUNK {chunk_num} =====\n\n")
                        
                        f.write("RAW CHUNK:\n")
                        f.write("-" * 40 + "\n")
                        f.write(chunk['raw'])
                        f.write("\n\n")
                        
                        f.write("PROCESSED CHUNK:\n")
                        f.write("-" * 40 + "\n")
                        f.write(chunk['processed'])
                        f.write("\n\n")
                        
                        f.write("-" * 80 + "\n\n")
                
                # Write raw transcription
                f.write("COMPLETE RAW TRANSCRIPTION\n")
                f.write("-" * 80 + "\n\n")
                for segment in segments:
                    f.write(f"[{format_time(segment['start'])} -> {format_time(segment['end'])}]\n")
                    f.write(f"{segment['text']}\n\n")
                
                # Write processed content
                f.write("\nCOMPLETE PROCESSED CONTENT\n")
                f.write("-" * 80 + "\n\n")
                f.write(processed_content)
            
            logging.info(f"Created full report at {report_path}")
            return report_path
            
        except Exception as e:
            logging.error(f"Error creating full report: {str(e)}")
            return None

class SystemState:
    """Manages global system state and configuration"""
    
    def __init__(self, base_dir: Path):
        # Initialize base paths
        self.base_dir = base_dir
        self.temp_dir = base_dir / "temp"
        self.output_dir = base_dir / "output"
        self.checkpoint_dir = base_dir / "checkpoints"
        
        # Create required directories
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.progress_tracker = ProgressTracker()
        self.checkpoint_manager = TranscriptionCheckpoint(self.checkpoint_dir)
        self.file_validator = FileValidator()
        self.output_manager = TranscriptionOutput(self.output_dir)
        
        # State tracking
        self.active_processes: Dict[str, asyncio.Task] = {}
        self.should_stop = False
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            # Remove temporary files
            for item in self.temp_dir.glob("*"):
                if item.is_file():
                    item.unlink()
            
            # Save final progress
            self.progress_tracker.save_progress()
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

class FileProcessor:
    """Handles file processing workflow and state management"""
    
    def __init__(self, system_state: SystemState, model: EnhancedWhisperModel, text_processor: TextProcessor):
        self.system_state = system_state
        self.model = model
        self.text_processor = text_processor
        self.max_retries = 3
        self.retry_delay = 30
        self.chunk_overlap = 0.5
    
    async def prepare_file(self, file_path: Path) -> Optional[Path]:
        """Prepare file for processing with enhanced validation"""
        try:
            # Validate file existence and size
            if not file_path.exists():
                logging.error(f"File not found: {file_path}")
                return None
                
            if file_path.stat().st_size == 0:
                logging.error(f"Empty file: {file_path}")
                return None
            
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                logging.error(f"File not readable: {file_path}")
                return None
            
            # Validate file format
            if not self.system_state.file_validator.is_supported_file(file_path):
                logging.error(f"Unsupported file format: {file_path}")
                return None
            
            # Get detailed file info
            audio_info = self.system_state.file_validator.get_audio_info(file_path)
            if not audio_info:
                logging.error(f"Could not get audio info for {file_path}")
                return None
            
            # Log file details
            logging.info(f"Processing file: {file_path}")
            logging.info(f"Duration: {audio_info['duration']:.2f}s")
            logging.info(f"Sample rate: {audio_info.get('sample_rate', 'unknown')} Hz")
            logging.info(f"Channels: {audio_info.get('channels', 'unknown')}")
            
            # Convert to WAV if needed
            if file_path.suffix.lower() != '.wav':
                logging.info(f"Converting {file_path} to WAV format")
                wav_path = self.system_state.file_validator.convert_to_wav(
                    file_path,
                    self.system_state.temp_dir
                )
                if not wav_path:
                    logging.error(f"Failed to convert {file_path} to WAV")
                    return None
                    
                # Verify conversion
                if not wav_path.exists() or wav_path.stat().st_size == 0:
                    logging.error("WAV conversion produced invalid file")
                    return None
                    
                return wav_path
            
            return file_path
            
        except Exception as e:
            logging.error(f"Error preparing file {file_path}: {str(e)}")
            return None
    
    async def transcribe_with_progress(
        self,
        file_path: Path,
        progress: TranscriptionProgress,
        start_position: float = 0,
        existing_segments: List[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Transcribe a file with progress tracking and checkpointing.
        
        Args:
            file_path (Path): Path to the audio file to transcribe.
            progress (TranscriptionProgress): Progress tracking object.
            start_position (float, optional): Starting position in seconds. Defaults to 0.
            existing_segments (List[Dict[str, Any]], optional): Existing segments. Defaults to None.
        
        Returns:
            Optional[List[Dict[str, Any]]]: List of transcribed segments if successful, None otherwise.
        """
        try:
            # Initialize segments with existing segments or an empty list
            segments = existing_segments or []
            
            # Transcribe
            new_segments = await self.model.transcribe_file(file_path, progress)
            if not new_segments:
                return None
            
            # Filter and combine segments
            for segment in new_segments:
                if segment['start'] >= start_position:
                    segments.append(segment)
            
            # Sort segments by start time
            segments.sort(key=lambda x: x['start'])
            
            # Save checkpoint
            self.system_state.checkpoint_manager.save_checkpoint(
                progress.filename,
                segments,
                progress.current_position
            )
            
            return segments
            
        except Exception as e:
            logging.error(f"Error in transcribe_with_progress: {str(e)}")
            return None
    
    async def process_file(self, file_path: Path) -> bool:
        """
        Process a single file with progress tracking and error handling.
        
        Args:
            file_path (Path): Path to the file to process.
        
        Returns:
            bool: True if processing is successful, False otherwise.
        """
        original_filename = file_path.name
        
        try:
            # Check if file is already completed
            if self.system_state.progress_tracker.is_completed(original_filename):
                logging.info(f"Skipping completed file: {original_filename}")
                return True
            
            # Initialize progress tracking
            progress = self.system_state.progress_tracker.initialize_progress(original_filename)
            progress.current_action = "Preparing file"
            self.system_state.progress_tracker.update_progress(
                original_filename,
                current_action=progress.current_action
            )
            
            # Prepare file
            prepared_path = await self.prepare_file(file_path)
            if not prepared_path:
                return False
            
            # Load checkpoint if exists
            checkpoint = self.system_state.checkpoint_manager.load_checkpoint(original_filename)
            start_position = checkpoint['last_position'] if checkpoint else 0
            existing_segments = checkpoint['segments'] if checkpoint else []
            
            # Update progress
            progress.current_action = "Transcribing audio"
            progress.current_position = start_position
            self.system_state.progress_tracker.update_progress(
                original_filename,
                current_action=progress.current_action,
                current_position=start_position
            )
            
            # Transcribe file
            segments = await self.transcribe_with_progress(
                prepared_path,
                progress,
                start_position,
                existing_segments
            )
            
            if not segments:
                return False
            
            # Process transcribed text
            progress.current_action = "Processing transcription"
            self.system_state.progress_tracker.update_progress(
                original_filename,
                current_action=progress.current_action
            )
            
            # Combine segments into text
            full_text = "\n".join(segment['text'] for segment in segments)
            
            # Process text
            processed_text = await self.text_processor.process_text(full_text, progress)
            if not processed_text:
                return False
            
            # Save outputs
            progress.current_action = "Saving outputs"
            self.system_state.progress_tracker.update_progress(
                original_filename,
                current_action=progress.current_action
            )
            
            # Save raw transcription
            raw_path = self.system_state.output_manager.save_raw_transcription(
                original_filename,
                segments
            )
            
            # Save processed output
            processed_path = self.system_state.output_manager.save_processed_output(
                original_filename,
                processed_text
            )
            
            # Get chunk results for the report
            chunk_results = self.text_processor.chunk_results.get(original_filename, {})
            
            # Create full report with chunk comparison
            report_path = self.system_state.output_manager.create_full_report(
                original_filename,
                segments,
                processed_text,
                chunk_results
            )
            
            # Mark as completed
            self.system_state.progress_tracker.mark_completed(original_filename)
            self.system_state.checkpoint_manager.delete_checkpoint(original_filename)
            
            # Clean up
            if prepared_path != file_path:
                prepared_path.unlink()
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing file {original_filename}: {str(e)}")
            return False

class ProcessManager:
    """Manages concurrent file processing and resource allocation"""
    
    def __init__(self, system_state: SystemState, max_concurrent: int = 1):
        self.system_state = system_state
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks: Dict[str, asyncio.Task] = {}
    
    async def process_file_with_semaphore(self, processor: FileProcessor, file_path: Path):
        """Process a file with resource limiting"""
        async with self.semaphore:
            try:
                await processor.process_file(file_path)
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
    
    async def process_directory(self, processor: FileProcessor, directory: Path):
        """Process all files in directory with concurrency control"""
        try:
            # Find all supported files
            files = []
            for ext in FileValidator.SUPPORTED_EXTENSIONS:
                files.extend(directory.glob(f"*{ext}"))
            
            if not files:
                logging.warning(f"No supported files found in {directory}")
                return
            
            # Update total files count
            total_files = len(files)
            logging.info(f"Found {total_files} files to process")
            
            # Create progress display
            with self.system_state.progress_tracker.progress:
                # Add overall progress
                overall_task = self.system_state.progress_tracker.progress.add_task(
                    "[cyan]Overall Progress",
                    total=total_files
                )
                
                # Process files
                tasks = []
                for file_path in files:
                    if self.system_state.should_stop:
                        break
                        
                    task = asyncio.create_task(
                        self.process_file_with_semaphore(processor, file_path)
                    )
                    tasks.append(task)
                    self.active_tasks[file_path.name] = task
                
                # Wait for all tasks
                completed = 0
                for task in asyncio.as_completed(tasks):
                    try:
                        await task
                        completed += 1
                        self.system_state.progress_tracker.progress.update(
                            overall_task,
                            completed=completed
                        )
                    except Exception as e:
                        logging.error(f"Task error: {str(e)}")
                
            logging.info("Directory processing completed")
            
        except Exception as e:
            logging.error(f"Error in process_directory: {str(e)}")
            raise
        finally:
            # Cleanup
            for task in self.active_tasks.values():
                if not task.done():
                    task.cancel()
            self.active_tasks.clear()

@dataclass
class ApplicationConfig:
    """Application configuration with validation"""
    
    # Path configuration
    input_dir: str = "input"
    output_dir: str = "output"
    temp_dir: str = "temp"
    checkpoint_dir: str = "checkpoints"
    
    # API configuration
    claude_api_key: str = ""
    
    # Model configuration
    model_name: str = "ivrit-ai/whisper-large-v3-turbo-ct2"
    compute_type: str = "float16"
    
    # Processing configuration  
    max_concurrent_files: int = 1
    chunk_size: int = 2000
    overlap_size: int = 100
    max_retries: int = 3
    retry_delay: int = 30
    
    # Language configuration
    source_language: str = "he"  # Hebrew
    target_language: str = "he"  # Hebrew
    
    # Claude configuration
    claude_model: str = "claude-3-5-sonnet-20241022"
    claude_max_tokens: int = 8192
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate paths
        for path_attr in ['output_dir', 'temp_dir', 'checkpoint_dir']:
            path = getattr(self, path_attr)
            if not isinstance(path, str):
                raise ValueError(f"{path_attr} must be a string")
        
        # Validate numerical values  
        if self.max_concurrent_files < 1:
            raise ValueError("max_concurrent_files must be at least 1")
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100")
        if self.overlap_size >= self.chunk_size:
            raise ValueError("overlap_size must be less than chunk_size")
        if self.max_retries < 1:  
            raise ValueError("max_retries must be at least 1")
        if self.retry_delay < 1:
            raise ValueError("retry_delay must be at least 1")
        
        # Set default input_dir if not provided or empty
        if not self.input_dir:
            script_dir = Path(__file__).parent  # Get script directory
            self.input_dir = str(script_dir / "input")  # Create path to input directory
            # Create input directory if it doesn't exist
            Path(self.input_dir).mkdir(exist_ok=True)
            logging.info(f"Using default input directory: {self.input_dir}")

class TranscriptionSystem:
    """Main transcription system orchestrator"""
    
    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.base_dir = Path.cwd()
        
        # Initialize system state
        self.system_state = SystemState(self.base_dir)
        
        # Initialize components
        self.model = None
        self.text_processor = None
        self.file_processor = None  
        self.process_manager = None
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize model
            self.model = EnhancedWhisperModel(
                self.config.model_name,
                self.system_state.progress_tracker
            )
            
            # Initialize text processor if API key is provided
            if self.config.claude_api_key:
                self.text_processor = TextProcessor(
                    self.config.claude_api_key,
                    self.system_state.progress_tracker  
                )
            else:
                logging.info("No Claude API key provided. Skipping text processing.")
            
            # Initialize file processor
            self.file_processor = FileProcessor(
                self.system_state,
                self.model,
                self.text_processor
            )
            
            # Initialize process manager  
            self.process_manager = ProcessManager(
                self.system_state,
                self.config.max_concurrent_files
            )
            
            logging.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing components: {str(e)}")
            return False
    
    def handle_interrupt(self, signum, frame):
        """Handle interrupt signals gracefully"""
        logging.info("\nReceived interrupt signal. Initiating graceful shutdown...")
        self.system_state.should_stop = True
    
    def cleanup(self):
        """Clean up system resources"""  
        try:
            if self.system_state:
                self.system_state.cleanup()
            
            # Clear CUDA memory if used
            if torch.cuda.is_available():  
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            gc.collect()
            logging.info("Cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
    
    async def run(self):
        """Run the transcription system"""
        try:
            # Initialize components
            if not self.initialize_components():
                return False
            
            # Process input directory
            input_dir = Path(self.config.input_dir)
            if not input_dir.exists():
                logging.error(f"Input directory not found: {input_dir}")
                return False
            
            # Display starting message
            console.print(Panel(
                "[bold green]Transcription System Starting[/bold green]\n\n"
                f"Input Directory: {input_dir}\n"
                f"Output Directory: {self.system_state.output_dir}\n"
                f"Model: {self.config.model_name}\n"
                f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
            ))
            
            # Start processing
            await self.process_manager.process_directory(
                self.file_processor,
                input_dir
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Error running transcription system: {str(e)}")
            return False
        finally:
            self.cleanup()

async def main():
    """Main entry point for the transcription system"""
    try:
        # Get input directory from user
        input_dir = input("Enter the input directory path (leave blank to use default 'input' directory): ").strip()
        # If input_dir is empty, it will use the default from ApplicationConfig

        # Get Claude API key from user (optional)
        api_key = input("Enter the Claude API key (leave blank to skip text processing): ")
        
        # Load configuration
        model_name = input("Enter Whisper model name (leave blank for default 'ivrit-ai/whisper-large-v3-turbo-ct2'): ").strip()
        config = ApplicationConfig(
            input_dir=input_dir if input_dir else "",
            claude_api_key=api_key,
            model_name=model_name if model_name else "ivrit-ai/whisper-large-v3-turbo-ct2"
)
        
        # Prompt user for custom Claude prompt (optional)
        if api_key:
            print("\nDefault Claude prompt:")
            print(TextProcessor.CLAUDE_PROMPT)
            choice = input("Enter 'v' to use the default prompt or 'new' to provide a custom one: ")
            if choice.lower() == 'new':
                custom_prompt = input("Enter the custom Claude prompt: ")
                TextProcessor.CLAUDE_PROMPT = custom_prompt
        
        # Create and run system
        system = TranscriptionSystem(config)
        success = await system.run()
        
        if success:
            console.print("[bold green]Transcription completed successfully![/bold green]")
        else:
            console.print("[bold red]Transcription completed with errors.[/bold red]")
            
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        console.print(f"[bold red]Critical error: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    # Set up asyncio event loop with error handling
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(main())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        sys.exit(0)  
    except Exception as e:
        console.print(f"[bold red]Fatal error: {str(e)}[/bold red]")
        sys.exit(1)
