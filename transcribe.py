#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcription System with Batch API Support
This script provides a comprehensive system for transcribing audio/video files
and processing the transcriptions using either regular Claude API or Batch API.

Author: Claude
Version: 2.0
"""

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
import requests
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from functools import wraps
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

# Anthropic API constants
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# Anthropic Batch API constants
ANTHROPIC_API_BATCH_URL = "https://api.anthropic.com/v1/messages/batches"
ANTHROPIC_API_BATCH_STATUS_URL = "https://api.anthropic.com/v1/messages/batches/{batch_id}"
ANTHROPIC_API_BATCH_RESULTS_URL = "https://api.anthropic.com/v1/messages/batches/{batch_id}/outputs"

# Max retries for API calls
MAX_API_RETRIES = 3
API_RETRY_DELAY = 5  # seconds

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
        """
        Calculate and update the estimated time to completion based on current progress.
        Updates the eta_seconds and estimated_completion_time attributes.
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Only calculate ETA if we have processed chunks and know the total
        if self.chunks_processed > 0 and self.total_chunks > 0:
            chunks_remaining = self.total_chunks - self.current_chunk
            
            # Calculate average time per chunk (need to handle division by zero)
            avg_time_per_chunk = elapsed_time / max(self.chunks_processed, 1)
            
            # Estimate remaining time
            self.eta_seconds = avg_time_per_chunk * chunks_remaining
            
            # Calculate estimated completion time
            self.estimated_completion_time = datetime.now() + timedelta(seconds=self.eta_seconds)
        elif self.total_chunks > 0:
            # If we know total but haven't processed any, use a default assumption
            self.eta_seconds = 60 * self.total_chunks  # Assume 1 minute per chunk as default
            self.estimated_completion_time = datetime.now() + timedelta(seconds=self.eta_seconds)
            
        self.last_update_time = current_time

@dataclass
class BatchChunk:
    """Represents a single chunk in a batch"""
    chunk_number: int
    custom_id: str
    content: str
    processed: Optional[str] = None
    thinking: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert batch chunk to dictionary for JSON serialization"""
        return {
            'chunk_number': self.chunk_number,
            'custom_id': self.custom_id,
            'content': self.content,
            'processed': self.processed,
            'thinking': self.thinking
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BatchChunk':
        """Create batch chunk from dictionary"""
        return cls(**data)

@dataclass
class BatchInfo:
    """Information about a batch for a single file"""
    filename: str
    batch_id: Optional[str] = None
    sent_time: Optional[datetime] = None
    status: str = "pending"
    error_message: Optional[str] = None
    retry_count: int = 0
    chunks: List[BatchChunk] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert batch info to dictionary for JSON serialization"""
        return {
            'filename': self.filename,
            'batch_id': self.batch_id,
            'sent_time': self.sent_time.isoformat() if self.sent_time else None,
            'status': self.status,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'chunks': [chunk.to_dict() for chunk in self.chunks]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BatchInfo':
        """Create batch info from dictionary"""
        chunks_data = data.pop('chunks', [])
        batch_info = cls(**data)
        
        if 'sent_time' in data and data['sent_time']:
            batch_info.sent_time = datetime.fromisoformat(data['sent_time'])
            
        batch_info.chunks = [BatchChunk.from_dict(chunk) for chunk in chunks_data]
        return batch_info


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

class BatchTracker:
    """Manages and tracks batch processing state"""
    
    def __init__(self):
        """Initialize the batch tracker"""
        self.batches: Dict[str, BatchInfo] = {}
        self.batch_file = Path("batch_tracking.json")
        self.load_state()
    
    def initialize_batch(self, filename: str, chunks: List[str]) -> BatchInfo:
        """
        Initialize a new batch for a file with chunks
        
        Args:
            filename (str): Name of the file being processed
            chunks (List[str]): List of text chunks to be processed
            
        Returns:
            BatchInfo: The initialized batch information
        """
        batch_info = BatchInfo(filename=filename)
        
        # Create batch chunks with custom IDs
        for i, chunk_content in enumerate(chunks, 1):
            custom_id = f"{filename}_chunk_{i}"
            batch_chunk = BatchChunk(
                chunk_number=i,
                custom_id=custom_id,
                content=chunk_content
            )
            batch_info.chunks.append(batch_chunk)
        
        # Store batch info
        self.batches[filename] = batch_info
        self.save_state()
        
        return batch_info
    
    def update_batch_submission(self, filename: str, batch_id: str) -> None:
        """
        Update batch with submission information
        
        Args:
            filename (str): Name of the file
            batch_id (str): Batch ID received from API
        """
        if filename in self.batches:
            # Validate batch_id format before saving
            if not isinstance(batch_id, str) or not batch_id.startswith("msgbatch_"):
                logging.error(f"Invalid batch_id format received: {batch_id}")
                self.batches[filename].error_message = f"Invalid batch_id format: {batch_id[:50]}..."
                self.batches[filename].status = "error"
            else:
                self.batches[filename].batch_id = batch_id
                self.batches[filename].sent_time = datetime.now()
                self.batches[filename].status = "submitted"
            self.save_state()
    
    def update_batch_status(self, filename: str, status: str, error_message: Optional[str] = None) -> None:
        """
        Update batch status
        
        Args:
            filename (str): Name of the file
            status (str): New status (pending, submitted, in_progress, completed, error)
            error_message (Optional[str]): Error message if status is error
        """
        if filename in self.batches:
            self.batches[filename].status = status
            if error_message:
                self.batches[filename].error_message = error_message
            self.save_state()
    
    def update_chunk_result(self, filename: str, custom_id: str, processed_content: str, thinking_content: Optional[str] = None) -> None:
        """
        Update a chunk with processed content and optional thinking content
        
        Args:
            filename (str): Name of the file
            custom_id (str): Custom ID of the chunk
            processed_content (str): Processed content of the chunk
            thinking_content (Optional[str]): Extended thinking content if available
        """
        if filename in self.batches:
            for chunk in self.batches[filename].chunks:
                if chunk.custom_id == custom_id:
                    # Always update processed content
                    chunk.processed = processed_content
                    
                    # Handle thinking content with more flexibility
                    if thinking_content:
                        # Either add as new attribute or update existing
                        if not hasattr(chunk, 'thinking'):
                            setattr(chunk, 'thinking', thinking_content)
                        else:
                            chunk.thinking = thinking_content
                    break
            
            # Save state after update
            self.save_state()
            
            # Log success
            logging.info(f"Updated chunk result for {filename}, chunk {custom_id}: {len(processed_content)} characters")

    def increment_retry_count(self, filename: str) -> int:
        """
        Increment retry count for a batch and return new count
        
        Args:
            filename (str): Name of the file
            
        Returns:
            int: New retry count
        """
        if filename in self.batches:
            self.batches[filename].retry_count += 1
            self.save_state()
            return self.batches[filename].retry_count
        return 0
    
    def get_pending_batches(self) -> List[BatchInfo]:
        """
        Get all batches that are pending or in progress
        
        Returns:
            List[BatchInfo]: List of pending or in-progress batches
        """
        return [
            batch for batch in self.batches.values() 
            if batch.status in ["pending", "submitted", "in_progress"]
        ]
    
    def get_batch_info(self, filename: str) -> Optional[BatchInfo]:
        """
        Get batch info for a specific file
        
        Args:
            filename (str): Name of the file
            
        Returns:
            Optional[BatchInfo]: Batch information or None if not found
        """
        return self.batches.get(filename)
    
    def is_batch_completed(self, filename: str) -> bool:
        """
        Check if a batch is completed
        
        Args:
            filename (str): Name of the file
            
        Returns:
            bool: True if batch is completed, False otherwise
        """
        return filename in self.batches and self.batches[filename].status == "completed"
    
    def is_batch_failed(self, filename: str) -> bool:
        """
        Check if a batch has failed
        
        Args:
            filename (str): Name of the file
            
        Returns:
            bool: True if batch has failed, False otherwise
        """
        return filename in self.batches and self.batches[filename].status == "error"
    
    def get_all_chunk_results(self, filename: str) -> List[Tuple[int, str]]:
        """
        Get all processed chunk results for a file in order
        
        Args:
            filename (str): Name of the file
            
        Returns:
            List[Tuple[int, str]]: List of tuples containing chunk number and processed content
        """
        if filename not in self.batches:
            return []
        
        # Get chunks, sort by number, and return only the processed content
        chunks = sorted(self.batches[filename].chunks, key=lambda x: x.chunk_number)
        return [(chunk.chunk_number, chunk.processed) for chunk in chunks if chunk.processed]
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about batches
        
        Returns:
            Dict[str, Any]: Dictionary containing batch statistics
        """
        total = len(self.batches)
        completed = sum(1 for b in self.batches.values() if b.status == "completed")
        failed = sum(1 for b in self.batches.values() if b.status == "error")
        pending = total - completed - failed
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'pending': pending
        }
    
    def save_state(self) -> None:
        """Save current state to JSON file"""
        try:
            data = {
                'batches': {
                    filename: batch.to_dict() 
                    for filename, batch in self.batches.items()
                },
                'last_update': datetime.now().isoformat()
            }
            
            with open(self.batch_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Error saving batch tracking state: {str(e)}")
    
    def load_state(self) -> None:
        """Load state from JSON file if it exists"""
        try:
            if self.batch_file.exists():
                with open(self.batch_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load batches
                if 'batches' in data:
                    self.batches = {
                        filename: BatchInfo.from_dict(batch_data)
                        for filename, batch_data in data['batches'].items()
                    }
                
                logging.info(f"Loaded batch tracking state: {len(self.batches)} batches")
                
        except Exception as e:
            logging.error(f"Error loading batch tracking state: {str(e)}")
            self.batches = {}

class ProgressTracker:
    """Manages progress tracking and reporting for the transcription process"""
    def __init__(self):
        """Initialize the progress tracker"""
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
        self.batch_progress = None  # Added for batch processing
    
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
        """
        Get duration of audio/video file using ffmpeg
        
        Args:
            file_path (Path): Path to the audio/video file
            
        Returns:
            float: Duration of the file in seconds
        """
        try:
            probe = ffmpeg.probe(str(file_path))
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            logging.error(f"Error getting duration for {file_path}: {str(e)}")
            return 0.0
    
    def initialize_progress(self, filename: str) -> TranscriptionProgress:
        """
        Initialize progress tracking for a file
        
        Args:
            filename (str): Name of the file
            
        Returns:
            TranscriptionProgress: Progress tracking object
        """
        progress = TranscriptionProgress(filename=filename)
        self.active_progresses[filename] = progress
        return progress
    
    def update_progress(self, filename: str, **kwargs):
        """
        Update progress for a specific file
        
        Args:
            filename (str): Name of the file
            **kwargs: Attributes to update on the progress object
        """
        if filename in self.active_progresses:
            progress = self.active_progresses[filename]
            for key, value in kwargs.items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
            progress.update_eta()
            self.save_progress()
    
    def mark_completed(self, filename: str):
        """
        Mark a file as completed
        
        Args:
            filename (str): Name of the file
        """
        self.completed_files.add(filename)
        if filename in self.active_progresses:
            del self.active_progresses[filename]
        self.save_progress()
    
    def is_completed(self, filename: str) -> bool:
        """
        Check if a file has been completed
        
        Args:
            filename (str): Name of the file
            
        Returns:
            bool: True if file has been completed, False otherwise
        """
        return filename in self.completed_files
    
    def get_progress(self, filename: str) -> Optional[TranscriptionProgress]:
        """
        Get progress for a specific file
        
        Args:
            filename (str): Name of the file
            
        Returns:
            Optional[TranscriptionProgress]: Progress object or None if not found
        """
        return self.active_progresses.get(filename)
    
    def start_batch_tracking(self, total_batches: int) -> int:
        """
        Start tracking batch processing progress
        
        Args:
            total_batches (int): Total number of batches to process
            
        Returns:
            int: ID of the batch progress task
        """
        self.batch_progress = self.progress.add_task(
            "[cyan]Batch Processing",
            total=total_batches
        )
        return self.batch_progress
    
    def update_batch_progress(self, completed: int):
        """
        Update batch processing progress
        
        Args:
            completed (int): Number of completed batches
        """
        if self.batch_progress is not None:
            self.progress.update(self.batch_progress, completed=completed)

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
        """
        Check if file format is supported
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            bool: True if file format is supported, False otherwise
        """
        return file_path.suffix.lower() in FileValidator.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def convert_to_wav(input_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Convert audio/video file to WAV format using ffmpeg
        
        Args:
            input_path (Path): Path to the input file
            output_dir (Path): Directory to save the output file
            
        Returns:
            Optional[Path]: Path to the output WAV file or None if conversion failed
        """
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
        """
        Get audio file information using ffmpeg
        
        Args:
            file_path (Path): Path to the audio file
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing audio information or None if failed
        """
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
        
class EnhancedWhisperModel:
    """Enhanced Whisper model wrapper with progress tracking and error handling"""
    
    def __init__(self, model_name: str, progress_tracker: 'ProgressTracker'):
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
        """
        Initialize the checkpoint manager
        
        Args:
            checkpoint_dir (Path): Directory to store checkpoint files
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.current_checkpoint = None
    
    def save_checkpoint(self, filename: str, segments: List[Dict[str, Any]], position: float):
        """
        Save transcription checkpoint
        
        Args:
            filename (str): Name of the file
            segments (List[Dict[str, Any]]): List of transcribed segments
            position (float): Current position in seconds
        """
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
        """
        Load transcription checkpoint if exists
        
        Args:
            filename (str): Name of the file
            
        Returns:
            Optional[Dict[str, Any]]: Checkpoint data or None if not found
        """
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
        """
        Delete transcription checkpoint
        
        Args:
            filename (str): Name of the file
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{filename}.checkpoint.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logging.info(f"Deleted checkpoint for {filename}")
                
        except Exception as e:
            logging.error(f"Error deleting checkpoint: {str(e)}")

class TranscriptionProgress:
    """
    Tracks the progress of a transcription and processing job.
    Provides serialization functionality and ETA calculation.
    """
    def __init__(self, filename: str):
        # Basic progress information
        self.filename = filename
        self.current_chunk = 0
        self.total_chunks = 0
        self.error_count = 0
        self.current_action = "Initializing"
        
        # Time tracking for ETA calculation
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.chunks_processed = 0
        self.estimated_completion_time = None
        self.eta_seconds = None
        
    def update_eta(self) -> None:
        """
        Calculate and update the estimated time to completion based on current progress.
        Updates the eta_seconds and estimated_completion_time attributes.
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Only calculate ETA if we have processed chunks and know the total
        if self.chunks_processed > 0 and self.total_chunks > 0:
            chunks_remaining = self.total_chunks - self.current_chunk
            
            # Calculate average time per chunk
            avg_time_per_chunk = elapsed_time / self.chunks_processed
            
            # Estimate remaining time
            self.eta_seconds = avg_time_per_chunk * chunks_remaining
            
            # Calculate estimated completion time
            self.estimated_completion_time = datetime.now() + timedelta(seconds=self.eta_seconds)
            
        self.last_update_time = current_time
    
    def increment_chunks_processed(self) -> None:
        """
        Increment the count of processed chunks and update ETA.
        Should be called after successfully processing each chunk.
        """
        self.chunks_processed += 1
        self.update_eta()
        
    def to_dict(self) -> dict:
        """
        Convert the progress object to a dictionary for serialization.
        
        Returns:
            dict: Dictionary representation of the progress
        """
        return {
            "filename": self.filename,
            "current_chunk": self.current_chunk,
            "total_chunks": self.total_chunks,
            "error_count": self.error_count,
            "current_action": self.current_action,
            "start_time": self.start_time,
            "last_update_time": self.last_update_time,
            "chunks_processed": self.chunks_processed,
            "eta_seconds": self.eta_seconds,
            "estimated_completion_time": self.estimated_completion_time.isoformat() if self.estimated_completion_time else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TranscriptionProgress':
        """
        Create a TranscriptionProgress object from a dictionary.
        
        Args:
            data (dict): Dictionary containing progress data
            
        Returns:
            TranscriptionProgress: New instance with restored data
        """
        progress = cls(data.get("filename", "unknown"))
        progress.current_chunk = data.get("current_chunk", 0)
        progress.total_chunks = data.get("total_chunks", 0)
        progress.error_count = data.get("error_count", 0)
        progress.current_action = data.get("current_action", "Restored from dict")
        
        # Restore time tracking fields
        progress.start_time = data.get("start_time", time.time())
        progress.last_update_time = data.get("last_update_time", time.time())
        progress.chunks_processed = data.get("chunks_processed", 0)
        progress.eta_seconds = data.get("eta_seconds")
        
        # Parse estimated completion time if available
        est_completion_str = data.get("estimated_completion_time")
        if est_completion_str:
            try:
                progress.estimated_completion_time = datetime.fromisoformat(est_completion_str)
            except (ValueError, TypeError):
                progress.estimated_completion_time = None
        else:
            progress.estimated_completion_time = None
            
        return progress

class TextProcessor:
    """
    Handles text processing, chunking, and correction using Claude API or Batch API.
    Supports both regular sequential processing and batch processing modes.
    """
    
    # Class constants
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    CHUNK_MIN_SIZE: int = 1500 
    CHUNK_MAX_SIZE: int = 3000
    OVERLAP_SIZE: int = 100
    
    # New system prompt
    # Claude system prompt template
    CLAUDE_SYSTEM_PROMPT: str = """You are an expert content specialist tasked with transforming raw Hebrew lecture transcriptions into clear, well-structured summaries. Your goal is to create comprehensive, organized summaries that capture all information from the original text while improving readability and structure.
Here are the key guidelines for creating your summaries:
1. Content Preservation:
   - Capture EVERY piece of information from the original text
   - Maintain all references, names, and numbers
   - Preserve the exact meaning of concepts
2. Structure and Organization:
   - Organize content into logical sections with clear headers
   - Group related information under descriptive headers
   - Present cause-and-effect relationships clearly
3. Formatting and Presentation:
   - Use bullet points to break down complex ideas
   - Present information in a more readable format while preserving all details
   - Use clear, precise terminology
4. Markdown Formatting:
   Apply the following Markdown syntax:
   - # Main headers for primary topics
   - ## Subheaders for subtopics
   - - Bullet points for individual items
   -   - Sub-bullets for related details
   - 1. Numbered lists when sequence matters
   - **Bold text** for emphasis
   - > Blockquotes for important quotes or highlights
   - Tables for structured data, e.g.:
     | Column 1 | Column 2 | Column 3 |
     |----------|----------|----------|
     | Data 1   | Data 2   | Data 3   |
5. Hebrew Language Optimization:
   - Ensure proper Hebrew grammar and syntax
   - Use appropriate Hebrew terminology for academic and professional contexts
   - Maintain the nuances and idioms specific to Hebrew language
6. Consistency and Efficiency:
   - Maintain a consistent structure across all summary segments
   - Focus on concise yet comprehensive presentation of information
   - Ensure that each summary segment can stand alone while also fitting into the larger context
Summary Planning Process:
Before creating your final summary, work through the following steps inside <summary_planning> tags in your thinking block:
1. List key topics and subtopics from the transcribed text
2. Quote important sentences or phrases in Hebrew, preserving the original language
3. Outline the structure of the summary, including main headers and subheaders
4. Draft key points for each section, ensuring all information from the original text is captured
5. Review your plan for completeness and accuracy, making sure no information is lost
Output your final summary between <improved_text> tags, using the Markdown formatting specified above."""

# Claude prompt template
    CLAUDE_PROMPT: str = """This is part <chunk_number>{chunk_number}</chunk_number> out of <total_chunks>{total_chunks}</total_chunks> from a Hebrew lecture.
Here is the transcribed text to summarize:
<transcribed_text>
{transcribed_text}
</transcribed_text>
Please create a comprehensive summary in Hebrew using the guidelines and Markdown formatting specified in the system instructions. Your final output should consist only of the improved text and should not duplicate or rehash any of the work you did in the summary planning section."""
    
    def __init__(self, api_key: str, progress_tracker: Any, use_batch_api: bool = True, 
         use_thinking: bool = True, thinking_budget_tokens: int = 16000,
         use_extended_output: bool = False):
        """
        Initialize the TextProcessor with API key, progress tracker, and processing mode.
        
        Args:
            api_key (str): The Claude API key
            progress_tracker (Any): Instance of progress tracker (should have update_progress method)
            use_batch_api (bool, optional): Whether to use Batch API. Defaults to True.
            use_thinking (bool, optional): Whether to use extended thinking. Defaults to True.
            thinking_budget_tokens (int, optional): Token budget for extended thinking. Defaults to 16000.
            use_extended_output (bool, optional): Whether to use extended output (up to 128K tokens). Defaults to False.
        
        Raises:
            ValueError: If api_key is invalid
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("Invalid API key provided")
        
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self.progress_tracker = progress_tracker
        self.use_batch_api = use_batch_api
        
        # Extended thinking settings
        self.use_thinking = use_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        
        # Extended output settings
        self.use_extended_output = use_extended_output
        self.claude_max_tokens = 128000 if use_extended_output else 8192
        
        # Claude model parameters
        self.claude_model = "claude-3-7-sonnet-20250219"  # Use the correct model name
        
        # Store processed chunks for reporting
        self.chunk_results = {}
        
        # Additional headers for API requests
        self.headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01", 
            "content-type": "application/json"
        }
        
        # Add beta header for extended output if configured
        if self.use_extended_output:
            self.headers["anthropic-beta"] = "output-128k-2025-02-19"
        
        # Store results URL for batch processing
        self.results_url = None
        
        # For tracking rate limits and backoff
        self.rate_limit_data = {}

        # Log initialization details
        features = []
        if use_batch_api:
            features.append("Batch API")
        if use_thinking:
            features.append(f"Extended Thinking ({thinking_budget_tokens} tokens)")
        if use_extended_output:
            features.append(f"Extended Output ({self.claude_max_tokens} tokens)")
        
        logging.info(f"TextProcessor initialized with: {', '.join(features) or 'Standard API'} mode")

        # Adjust chunk sizes based on extended output capabilities
        if self.use_extended_output:
            # If using extended output, we can work with much larger chunks
            self.CHUNK_MAX_SIZE = 25000  # Increased from 20000 for better efficiency
            self.CHUNK_MIN_SIZE = 1   # Larger minimum threshold
            self.OVERLAP_SIZE = 100      # More overlap for context
            logging.info(f"Using extended chunk sizes: max={self.CHUNK_MAX_SIZE}, min={self.CHUNK_MIN_SIZE}, overlap={self.OVERLAP_SIZE}")
        else:
            # Default chunk sizes for regular output
            self.CHUNK_MAX_SIZE = 3000
            self.CHUNK_MIN_SIZE = 1
            self.OVERLAP_SIZE = 100  # Increased from 100 for better continuity

    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into appropriately sized chunks with comprehensive validation,
        optimizing for extended output when enabled.
        
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
            
            # Check if batch processing is suitable based on chunk count
            if self.use_batch_api and len(chunks) > 5 and not self.use_extended_output:
                logging.info(f"Split text into {len(chunks)} chunks - using batch processing")
            elif self.use_extended_output and len(chunks) <= 3:
                logging.info(f"Split text into {len(chunks)} large chunks with extended output capabilities")
            elif self.use_extended_output and len(chunks) > 3:
                logging.info(f"Split text into {len(chunks)} chunks with extended output - consider using batch processing for efficiency")
                
            return chunks
                
        except Exception as e:
            logging.error(f"Error in split_into_chunks: {str(e)}")
            return []

    async def process_chunk(self, chunk: str, chunk_number: int, total_chunks: int, progress: TranscriptionProgress) -> Optional[str]:
        """
        Process a single chunk with comprehensive error handling using regular API.
        Supports extended thinking for improved results.
        
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
                progress.update_eta()
                
                self.progress_tracker.update_progress(
                    progress.filename,
                    current_chunk=chunk_number,
                    total_chunks=total_chunks,
                    current_action=progress.current_action,
                    eta_seconds=progress.eta_seconds
                )
                
                # Validate input
                if not isinstance(chunk, str) or not chunk.strip():
                    raise ValueError("Invalid or empty chunk provided")

                # Prepare request based on format used in create_batch_request
                prompt_parts = [
                    {
                        "type": "text",
                        "text": self.CLAUDE_PROMPT.format(
                            chunk_number=chunk_number,
                            total_chunks=total_chunks,
                            transcribed_text=chunk
                        )
                    }
                ]
                
                # Prepare API request parameters
                request_params = {
                    "model": self.claude_model,
                    "max_tokens": self.claude_max_tokens,
                    "messages": [{"role": "user", "content": prompt_parts}],
                    "system": self.CLAUDE_SYSTEM_PROMPT
                }
                
                # Add extended thinking if enabled
                if self.use_thinking:
                    request_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.thinking_budget_tokens
                    }
                    logging.info(f"Using extended thinking with budget of {self.thinking_budget_tokens} tokens")
                
                # Add beta header for extended output if configured
                headers = self.headers.copy()
                if self.use_extended_output:
                    headers["anthropic-beta"] = "output-128k-2025-02-19"
                
                # Make API request through client
                if self.use_thinking or self.use_extended_output:
                    # Use raw API call for advanced features
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: requests.post(
                            ANTHROPIC_API_URL,
                            headers=headers,
                            json=request_params
                        )
                    )
                    
                    # Check status and extract rate limits
                    rate_limits = self.parse_rate_limit_headers(response)
                    
                    if response.status_code == 429:
                        retry_after = rate_limits.get('retry_after', self.RETRY_DELAY * attempt)
                        logging.warning(f"Rate limited! Waiting {retry_after} seconds before retry")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    elif response.status_code != 200:
                        logging.error(f"API error: {response.status_code}, {response.text[:200]}...")
                        await asyncio.sleep(self.RETRY_DELAY * attempt)
                        continue
                    
                    # Process response
                    response_data = response.json()
                    content_blocks = response_data.get('content', [])
                    
                    # Extract thinking and text blocks
                    thinking_blocks = []
                    text_content = ""
                    
                    for block in content_blocks:
                        if block.get('type') == 'thinking':
                            thinking_blocks.append(block.get('thinking', ''))
                        elif block.get('type') == 'redacted_thinking':
                            thinking_blocks.append("<redacted thinking block>")
                        elif block.get('type') == 'text':
                            text_content += block.get('text', '')
                    
                    # Log thinking if available
                    if thinking_blocks and self.use_thinking:
                        logging.info(f"Extended thinking produced {len(thinking_blocks)} thinking blocks")
                        
                        # Store thinking blocks for analysis
                        if progress.filename not in self.chunk_results:
                            self.chunk_results[progress.filename] = {}
                        
                        # Store with the chunk for later analysis
                        self.chunk_results[progress.filename][f"thinking_{chunk_number}"] = "\n\n".join(thinking_blocks)
                    
                    # Extract text between improved_text tags
                    match = re.search(r'<improved_text>\s*(.*?)(?=</improved_text>|$)', text_content, re.DOTALL | re.IGNORECASE)
                    if not match:
                        raise ValueError("Failed to find improved_text tags in response")
                    
                    improved_text = match.group(1).strip()
                    
                else:
                    # Use standard client for basic usage
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.client.messages.create(
                            model=self.claude_model,
                            max_tokens=self.claude_max_tokens,
                            messages=[{"role": "user", "content": prompt_parts}],
                            system="You are an AI assistant tasked with improving transcribed Hebrew content."
                        )
                    )
                    
                    # Extract content properly
                    if not hasattr(response, 'content') or not response.content:
                        raise ValueError("Response missing content")
                    
                    content = response.content[0].text if isinstance(response.content, list) else response.content
                    
                    if not isinstance(content, str):
                        raise TypeError(f"Unexpected content type: {type(content)}")

                    # Extract text between improved_text tags
                    match = re.search(r'<improved_text>\s*(.*?)(?=</improved_text>|$)', content, re.DOTALL | re.IGNORECASE)
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
                
                # Update progress to indicate successful processing
                progress.increment_chunks_processed()
                self.progress_tracker.update_progress(
                    progress.filename,
                    chunks_processed=progress.chunks_processed,
                    eta_seconds=progress.eta_seconds
                )
                
                return improved_text
                    
            except Exception as e:
                logging.error(f"Error processing chunk {chunk_number} (attempt {attempt}/{self.MAX_RETRIES}): {str(e)}")
                progress.error_count += 1  
                self.progress_tracker.update_progress(
                    progress.filename,
                    error_count=progress.error_count
                )
                
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.RETRY_DELAY * attempt)  # Exponential backoff
                    continue
                
                return None

    def create_batch_request(self, chunks: List[str], chunk_ids: List[str]) -> dict:
        """
        Create a batch request for processing multiple chunks with prompt caching support.
        Format follows Anthropic's Message Batches API specification.
        
        Args:
            chunks (List[str]): List of text chunks to process
            chunk_ids (List[str]): List of custom IDs for each chunk
            
        Returns:
            dict: Batch request in the format required by the Message Batches API
        """
        requests = []
        
        # Create system prompt - common across all requests
        system_prompt = self.CLAUDE_SYSTEM_PROMPT
        
        for i, (chunk, custom_id) in enumerate(zip(chunks, chunk_ids), 1):
            total_chunks = len(chunks)
            
            # Format the user prompt with current chunk information
            user_prompt = self.CLAUDE_PROMPT.format(
                chunk_number=i,
                total_chunks=total_chunks,
                transcribed_text=chunk
            )
            
            # Prepare prompt parts
            prompt_parts = [
                {
                    "type": "text",
                    "text": user_prompt
                    # No cache_control here as this contains unique content
                }
            ]
            
            # Create request input with updated format
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": self.claude_model,
                    "max_tokens": self.claude_max_tokens,
                    "messages": [
                        {"role": "user", "content": prompt_parts}
                    ],
                    "system": system_prompt
                }
            }
            
            # Add extended thinking if enabled
            if hasattr(self, 'use_thinking') and self.use_thinking:
                # Ensure thinking budget doesn't exceed what's recommended in docs
                thinking_budget = min(32000, self.thinking_budget_tokens)
                request['params']['thinking'] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }
            
            requests.append(request)
        
        return {
            "requests": requests
        }

    async def submit_batch(self, batch_request: dict) -> Optional[str]:
        """
        Submit a batch request to the Message Batches API
        
        Args:
            batch_request (dict): Batch request data
            
        Returns:
            Optional[str]: Batch ID if successful, None otherwise
        """
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Validate batch request structure before submission
                if not isinstance(batch_request, dict) or 'requests' not in batch_request:
                    logging.error(f"Invalid batch request structure: missing 'requests' key")
                    return None
                
                if not batch_request['requests'] or not isinstance(batch_request['requests'], list):
                    logging.error(f"Invalid batch request: 'requests' must be a non-empty list")
                    return None
                
                # Make API request to create batch using the correct endpoint
                url = "https://api.anthropic.com/v1/messages/batches"
                
                # Log the request structure for debugging (without full content)
                sample_request = {
                    "custom_id": batch_request['requests'][0].get('custom_id', 'N/A'),
                    "params": {
                        "model": batch_request['requests'][0].get('params', {}).get('model', 'N/A'),
                        "max_tokens": batch_request['requests'][0].get('params', {}).get('max_tokens', 'N/A'),
                        "messages": "[CONTENT OMITTED]"
                    }
                }
                logging.debug(f"Submitting batch request with {len(batch_request['requests'])} items. Sample request: {sample_request}")
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.post(
                        url,
                        headers=self.headers,
                        json=batch_request
                    )
                )
                
                # Check response status
                if response.status_code != 200:
                    logging.error(f"Batch submission failed with code {response.status_code}")
                    logging.error(f"Response: {response.text}")
                    
                    # Handle common errors with more specific messages
                    if response.status_code == 400:
                        try:
                            error_data = response.json()
                            error_message = error_data.get('error', {}).get('message', 'Unknown error')
                            logging.error(f"Batch submission validation error: {error_message}")
                            
                            # If it's a custom_id format error, check our sanitization
                            if "custom_id" in error_message and "pattern" in error_message:
                                logging.error("The custom_id format is invalid. Verify sanitization function.")
                        except:
                            pass
                    
                    # Explicitly don't raise the exception - we'll handle the error and retry or exit
                    if attempt < self.MAX_RETRIES:
                        await asyncio.sleep(self.RETRY_DELAY)
                        continue
                    return None
                
                # Parse the response
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON response from batch submission: {response.text[:100]}...")
                    if attempt < self.MAX_RETRIES:
                        await asyncio.sleep(self.RETRY_DELAY)
                        continue
                    return None
                
                # Extract batch ID
                batch_id = response_data.get('id')
                if not batch_id:
                    logging.error(f"No batch ID in response: {response_data}")
                    if attempt < self.MAX_RETRIES:
                        await asyncio.sleep(self.RETRY_DELAY)
                        continue
                    return None
                
                # Validate batch ID format
                if not isinstance(batch_id, str) or not batch_id.startswith("msgbatch_"):
                    logging.error(f"Invalid batch ID format received: {batch_id}")
                    if attempt < self.MAX_RETRIES:
                        await asyncio.sleep(self.RETRY_DELAY)
                        continue
                    return None
                
                logging.info(f"Successfully submitted batch, ID: {batch_id}")
                return batch_id
                
            except requests.exceptions.RequestException as re:
                logging.error(f"HTTP error submitting batch (attempt {attempt}/{self.MAX_RETRIES}): {str(re)}")
            except Exception as e:
                logging.error(f"Error submitting batch (attempt {attempt}/{self.MAX_RETRIES}): {str(e)}")
            
            # Log the batch request structure (without the full content for brevity)
            debug_batch = {
                "structure": "Batch request structure",
                "requests_count": len(batch_request.get("requests", [])),
                "first_custom_id": batch_request.get("requests", [{}])[0].get("custom_id", "N/A") if batch_request.get("requests") else "No requests"
            }
            logging.error(f"Batch request debug info: {debug_batch}")
            
            if attempt < self.MAX_RETRIES:
                await asyncio.sleep(self.RETRY_DELAY)
                continue
            
            return None

    async def check_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Check the status of a batch and return detailed stats
        
        Args:
            batch_id (str): Batch ID to check
            
        Returns:
            Optional[Dict[str, Any]]: Batch status information or None if failed
        """
        try:
            # Strict validation of batch_id format before making API call
            if not isinstance(batch_id, str):
                logging.error(f"Invalid batch_id type: {type(batch_id)}")
                return None
                    
            if not batch_id.startswith("msgbatch_"):
                logging.error(f"Invalid batch_id format: {batch_id[:50]}...")
                return None
            
            # Retrieve batch status using the correct endpoint
            url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(
                    url,
                    headers=self.headers
                )
            )
            
            # Parse rate limit headers
            rate_limits = self.parse_rate_limit_headers(response)
            
            # Check HTTP status code
            if response.status_code != 200:
                logging.error(f"Batch status check failed with code {response.status_code}: {response.text[:200]}...")
                
                # Handle rate limiting
                if response.status_code == 429 and rate_limits['retry_after']:
                    logging.warning(f"Rate limited! Need to wait {rate_limits['retry_after']} seconds")
                    return {"status": "rate_limited", "retry_after": rate_limits['retry_after']}
                    
                return None
                    
            # Parse response data
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response from batch status check: {response.text[:100]}...")
                return None
            
            # Extract batch status and request counts
            status = response_data.get('processing_status')
            request_counts = response_data.get('request_counts', {})
            
            if not status:
                logging.error("No processing_status field in response")
                return None
            
            # Log detailed statistics
            logging.info(f"Batch {batch_id} status: {status}")
            logging.info(f"Request counts: Processing: {request_counts.get('processing', 0)}, "
                    f"Succeeded: {request_counts.get('succeeded', 0)}, "
                    f"Errored: {request_counts.get('errored', 0)}, "
                    f"Canceled: {request_counts.get('canceled', 0)}, "
                    f"Expired: {request_counts.get('expired', 0)}")
                        
            # Store results URL for later retrieval
            results_url = response_data.get('results_url')
            if results_url:
                self.results_url = results_url
                logging.debug(f"Stored results_url: {results_url}")
            
            # Return comprehensive status info
            return {
                "status": status,
                "request_counts": request_counts,
                "results_url": results_url,
                "rate_limits": rate_limits,
                "raw_response": response_data
            }
                
        except requests.exceptions.RequestException as re:
            logging.error(f"HTTP error checking batch status for {batch_id}: {str(re)}")
            return None
        except Exception as e:
            logging.error(f"Error checking batch status for {batch_id}: {str(e)}")
            return None

    async def retrieve_batch_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve results from a completed batch with support for extended thinking results.
        
        Args:
            batch_id (str): Batch ID to retrieve results for
            
        Returns:
            Optional[Dict[str, Any]]: Dictionary mapping custom_ids to processed text
        """
        try:
            if not hasattr(self, 'results_url') or not self.results_url:
                # If results_url isn't available, fetch it first
                url = f"https://api.anthropic.com/v1/messages/batches/{batch_id}"
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.get(
                        url,
                        headers=self.headers
                    )
                )
                response.raise_for_status()
                response_data = response.json()
                self.results_url = response_data.get('results_url')
                
                if not self.results_url:
                    raise ValueError("No results URL available")
            
            # Retrieve batch results using the results URL
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(
                    self.results_url,
                    headers=self.headers
                )
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Process rate limits
            rate_limits = self.parse_rate_limit_headers(response)
            
            # Stream the results
            results = {}  # Dictionary to store processed results by custom_id
            thinking_results = {}  # Store thinking blocks separately
            
            # Process the JSONL response line by line
            for line in response.iter_lines():
                if line:
                    try:
                        # Decode byte string to regular string
                        decoded_line = line.decode('utf-8')
                        result = json.loads(decoded_line)
                        
                        custom_id = result.get('custom_id')
                        if not custom_id:
                            logging.warning("Result missing custom_id")
                            continue
                            
                        result_data = result.get('result', {})
                        result_type = result_data.get('type')
                        
                        if result_type == "succeeded":
                            # Extract message content
                            message = result_data.get('message', {})
                            content_blocks = message.get('content', [])
                            
                            # Init containers for text and thinking content
                            text_content = ""
                            thinking_blocks = []
                            
                            # Process each content block
                            for block in content_blocks:
                                block_type = block.get('type')
                                
                                if block_type == 'thinking':
                                    thinking_blocks.append(block.get('thinking', ''))
                                elif block_type == 'redacted_thinking':
                                    thinking_blocks.append("<redacted thinking block>")
                                elif block_type == 'text':
                                    text_content += block.get('text', '')
                            
                            # If thinking blocks exist, store them separately
                            if thinking_blocks:
                                thinking_results[custom_id] = "\n\n".join(thinking_blocks)
                                logging.info(f"Retrieved {len(thinking_blocks)} thinking blocks for {custom_id}")
                            
                            # Extract improved text using regex - FIX HERE
                            if text_content:
                                match = re.search(r'<improved_text>\s*([\s\S]*?)(?=</improved_text>|$)', text_content, re.DOTALL)
                                
                                if match:
                                    improved_text = match.group(1).strip()
                                    results[custom_id] = improved_text
                                    logging.info(f"Successfully processed result for {custom_id}")
                                else:
                                    # Try a simpler approach if the regex fails
                                    if '<improved_text>' in text_content and '</improved_text>' in text_content:
                                        parts = text_content.split('<improved_text>', 1)[1]
                                        improved_text = parts.split('</improved_text>', 1)[0].strip()
                                        results[custom_id] = improved_text
                                        logging.info(f"Successfully processed result for {custom_id} using alternative method")
                                    else:
                                        logging.warning(f"Could not extract improved_text from response for {custom_id}")
                                        # Save the raw text as fallback
                                        results[custom_id] = text_content.strip()
                                        logging.info(f"Saved raw content as fallback for {custom_id}")
                            else:
                                logging.warning(f"No text content in response for {custom_id}")
                                
                        elif result_type in ["errored", "expired", "canceled"]:
                            error_msg = result_data.get('error', {}).get('message', 'Unknown error')
                            logging.warning(f"Request {custom_id} {result_type}: {error_msg}")
                        else:
                            logging.warning(f"Unknown result type for {custom_id}: {result_type}")
                    except json.JSONDecodeError:
                        logging.error(f"Failed to parse result line as JSON: {line}")
                    except Exception as e:
                        logging.error(f"Error processing result line: {str(e)}")
            
            # Store parsed results in chunk_results
            for custom_id, improved_text in results.items():
                # Parse filename and chunk number from custom_id (format: "{filename}_chunk_{chunk_number}")
                parts = custom_id.split('_chunk_')
                if len(parts) == 2:
                    filename = parts[0]
                    
                    if filename not in self.chunk_results:
                        self.chunk_results[filename] = {}
                    
                    self.chunk_results[filename][custom_id] = {
                        'processed': improved_text,
                        'thinking': thinking_results.get(custom_id)
                    }
            
            logging.info(f"Successfully retrieved and processed {len(results)} results for batch {batch_id}")
            return {
                'results': results,
                'thinking_blocks': thinking_results,
                'total_results': len(results),
                'rate_limits': rate_limits
            }
                
        except Exception as e:
            logging.error(f"Error retrieving batch results for {batch_id}: {str(e)}")
            logging.error(traceback.format_exc())
            return None

    async def process_text(self, text: str, progress: TranscriptionProgress) -> Optional[str]:
        """
        Process complete text with chunking and progress tracking, using either regular API or Batch API
        
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
            
            # Update progress with total chunks
            progress.total_chunks = len(chunks)
            progress.update_eta()
            self.progress_tracker.update_progress(
                progress.filename,
                total_chunks=len(chunks),
                eta_seconds=progress.eta_seconds
            )
            
            # Determine best processing strategy based on chunks size and count
            if self.use_extended_output and len(chunks) == 1:
                logging.info("Using direct processing with extended output for single chunk")
                # Use regular API processing for single extended chunk
                return await self._process_text_regular(chunks, progress)
            elif self.use_extended_output and len(chunks) <= 3 and not self.use_batch_api:
                logging.info("Using sequential processing with extended output for small number of chunks")
                return await self._process_text_regular(chunks, progress)
            elif self.use_batch_api:
                logging.info(f"Using batch processing for {len(chunks)} chunks")
                return await self._process_text_batch(chunks, progress)
            else:
                logging.info(f"Using regular sequential processing for {len(chunks)} chunks")
                return await self._process_text_regular(chunks, progress)
                
        except Exception as e:
            logging.error(f"Error in process_text: {str(e)}")
            return None

    async def _process_text_regular(self, chunks: List[str], progress: TranscriptionProgress) -> Optional[str]:
        """
        Process text using regular API (process each chunk sequentially)
        
        Args:
            chunks (List[str]): List of text chunks
            progress (TranscriptionProgress): Progress tracking object
            
        Returns:
            Optional[str]: Processed text if successful, None otherwise
        """
        try:
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
            logging.error(f"Error in _process_text_regular: {str(e)}")
            return None

    def sanitize_custom_id(self, input_id: str) -> str:
        """
        Sanitize custom ID to match API requirements (alphanumeric, underscores, hyphens only, max 64 chars).
        
        Args:
            input_id (str): The original custom ID
            
        Returns:
            str: Sanitized custom ID conforming to '^[a-zA-Z0-9_-]{1,64}' pattern
        """
        import re
        import hashlib
        
        # Ensure custom_id_mapping exists
        if not hasattr(self, 'custom_id_mapping'):
            self.custom_id_mapping = {}
        
        # Create a deterministic hash to ensure uniqueness, especially for non-ASCII characters
        id_hash = hashlib.md5(input_id.encode('utf-8')).hexdigest()[:10]
        
        # For Hebrew or other non-ASCII filenames, don't try to preserve the original characters
        # Just create a completely safe ID based on a prefix and the hash
        safe_id = f"file_{id_hash}"
        
        # Extract only allowed characters from the original ID
        # This won't preserve Hebrew but might keep English parts of the filename
        english_chars = re.sub(r'[^a-zA-Z0-9_-]', '', input_id)
        if english_chars:
            if len(english_chars) > 40:  # Limit length to leave room for hash
                english_chars = english_chars[:40]
            safe_id = f"{english_chars}_{id_hash}"
        
        # Final safety check - ensure the total length doesn't exceed 64 characters
        if len(safe_id) > 64:
            safe_id = f"file_{id_hash}"  # Fall back to shortest safe form
        
        # Final validation - the API is very strict about this format
        if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', safe_id):
            safe_id = f"file_{id_hash}"
        
        # Store mapping for later reference
        self.custom_id_mapping[safe_id] = input_id
        logging.debug(f"Sanitized ID: '{input_id}' → '{safe_id}'")
        
        return safe_id

    async def _process_text_batch(self, chunks: List[str], progress: TranscriptionProgress) -> Optional[str]:
        """
        Process text using Batch API (send all chunks in a batch)
        
        Args:
            chunks (List[str]): List of text chunks
            progress (TranscriptionProgress): Progress tracking object
            
        Returns:
            Optional[str]: Processed text if successful, None otherwise
        """
        try:
            # Create custom IDs for each chunk with proper sanitization for API requirements
            original_chunk_ids = [f"{progress.filename}_chunk_{i}" for i in range(1, len(chunks) + 1)]
            sanitized_chunk_ids = [self.sanitize_custom_id(cid) for cid in original_chunk_ids]
            
            # Create mapping between sanitized and original IDs
            id_mapping = dict(zip(sanitized_chunk_ids, original_chunk_ids))
            
            # Update progress
            progress.current_action = f"Submitting {len(chunks)} chunks for batch processing"
            progress.update_eta()
            self.progress_tracker.update_progress(
                progress.filename,
                current_action=progress.current_action,
                eta_seconds=progress.eta_seconds
            )
            
            # Create batch request with sanitized IDs
            batch_request = self.create_batch_request(chunks, sanitized_chunk_ids)
            
            # Submit batch
            batch_id = await self.submit_batch(batch_request)
            if not batch_id:
                logging.error("Batch submission failed, falling back to regular processing")
                progress.current_action = "Batch processing failed, using regular processing"
                self.progress_tracker.update_progress(
                    progress.filename,
                    current_action=progress.current_action
                )
                
                # Process using regular API instead
                result = await self._process_text_regular(chunks, progress)
                
                # Make sure to update progress after completion
                if result:
                    progress.current_action = "Completed regular processing (batch mode failed)"
                    progress.chunks_processed = len(chunks)
                    progress.update_eta()
                    self.progress_tracker.update_progress(
                        progress.filename,
                        current_action=progress.current_action,
                        chunks_processed=progress.chunks_processed
                    )
                
                return result
            
            # Additional verification that batch_id is valid
            if not isinstance(batch_id, str) or not batch_id.startswith("msgbatch_"):
                logging.error(f"Invalid batch_id received: {batch_id}")
                # Fall back to regular processing
                return await self._process_text_regular(chunks, progress)
            
            # Store raw chunks for report using original IDs
            if progress.filename not in self.chunk_results:
                self.chunk_results[progress.filename] = {}
            
            for i, chunk in enumerate(chunks, 1):
                original_id = original_chunk_ids[i-1]
                sanitized_id = sanitized_chunk_ids[i-1]
                self.chunk_results[progress.filename][original_id] = {
                    'raw': chunk,
                    'processed': None,  # Will be filled when batch completes
                    'sanitized_id': sanitized_id  # Store mapping to sanitized ID
                }
            
            # Update progress
            progress.current_action = f"Batch submitted successfully (ID: {batch_id})"
            progress.update_eta()
            self.progress_tracker.update_progress(
                progress.filename,
                current_action=progress.current_action,
                eta_seconds=progress.eta_seconds
            )
            
            # For immediate processing result, we return batch_id as a string
            # The actual results will be processed later by the batch tracker
            return batch_id
            
        except Exception as e:
            logging.error(f"Error in _process_text_batch: {str(e)}")
            logging.error(traceback.format_exc())
            # On exception, fall back to regular processing
            try:
                logging.info("Attempting fallback to regular processing due to exception in batch processing")
                return await self._process_text_regular(chunks, progress)
            except Exception as fallback_error:
                logging.error(f"Fallback to regular processing also failed: {str(fallback_error)}")
                return None

    def parse_rate_limit_headers(self, response: requests.Response) -> Dict[str, Any]:
        """
        Parse rate limit headers from response to manage request timing.
        
        Args:
            response (requests.Response): HTTP response from API call
            
        Returns:
            Dict[str, Any]: Dictionary containing rate limit information
        """
        rate_limits = {
            'retry_after': None,
            'requests': {
                'limit': None,
                'remaining': None,
                'reset': None
            },
            'tokens': {
                'limit': None,
                'remaining': None,
                'reset': None
            },
            'input_tokens': {
                'limit': None,
                'remaining': None,
                'reset': None
            },
            'output_tokens': {
                'limit': None,
                'remaining': None,
                'reset': None
            }
        }
        
        # Extract retry-after header
        if 'retry-after' in response.headers:
            try:
                rate_limits['retry_after'] = int(response.headers['retry-after'])
            except (ValueError, TypeError):
                pass
        
        # Extract other rate limit headers
        header_mapping = {
            'anthropic-ratelimit-requests-limit': ('requests', 'limit'),
            'anthropic-ratelimit-requests-remaining': ('requests', 'remaining'),
            'anthropic-ratelimit-requests-reset': ('requests', 'reset'),
            'anthropic-ratelimit-tokens-limit': ('tokens', 'limit'),
            'anthropic-ratelimit-tokens-remaining': ('tokens', 'remaining'),
            'anthropic-ratelimit-tokens-reset': ('tokens', 'reset'),
            'anthropic-ratelimit-input-tokens-limit': ('input_tokens', 'limit'),
            'anthropic-ratelimit-input-tokens-remaining': ('input_tokens', 'remaining'),
            'anthropic-ratelimit-input-tokens-reset': ('input_tokens', 'reset'),
            'anthropic-ratelimit-output-tokens-limit': ('output_tokens', 'limit'),
            'anthropic-ratelimit-output-tokens-remaining': ('output_tokens', 'remaining'),
            'anthropic-ratelimit-output-tokens-reset': ('output_tokens', 'reset')
        }
        
        for header, (category, field) in header_mapping.items():
            if header in response.headers:
                try:
                    if 'reset' in field:
                        # Parse datetime in RFC 3339 format
                        rate_limits[category][field] = datetime.fromisoformat(response.headers[header].replace('Z', '+00:00'))
                    else:
                        # Parse numerical values
                        rate_limits[category][field] = int(response.headers[header])
                except (ValueError, TypeError) as e:
                    logging.warning(f"Failed to parse header {header}: {e}")
        
        # Track overall usage
        self.rate_limit_data = {
            'last_updated': datetime.now(),
            'limits': rate_limits
        }
        
        # Log rate limit info if close to limits
        for category in ['requests', 'tokens', 'input_tokens', 'output_tokens']:
            limit = rate_limits[category]['limit']
            remaining = rate_limits[category]['remaining']
            
            if limit and remaining and limit > 0:
                usage_percent = (1 - (remaining / limit)) * 100
                if usage_percent > 80:
                    logging.warning(f"{category} usage at {usage_percent:.1f}% of limit")
        
        return rate_limits

    async def adaptive_rate_limit_backoff(self) -> int:
        """
        Calculate adaptive backoff time based on current rate limit data.
        
        Returns:
            int: Number of seconds to wait before next request
        """
        if not self.rate_limit_data or 'limits' not in self.rate_limit_data:
            return self.RETRY_DELAY
        
        # Get current rate limit data
        rate_limits = self.rate_limit_data['limits']
        
        # If we have a specific retry_after value, use it
        if rate_limits['retry_after']:
            return rate_limits['retry_after']
        
        # Calculate backoff based on remaining requests/tokens
        backoff_seconds = self.RETRY_DELAY
        
        # Check request limits
        if (rate_limits['requests']['limit'] and 
            rate_limits['requests']['remaining'] and 
            rate_limits['requests']['limit'] > 0):
            
            requests_ratio = rate_limits['requests']['remaining'] / rate_limits['requests']['limit']
            if requests_ratio < 0.1:  # Less than 10% remaining
                backoff_seconds = max(backoff_seconds, 30)
            elif requests_ratio < 0.2:  # Less than 20% remaining
                backoff_seconds = max(backoff_seconds, 15)
        
        # Check token limits 
        if (rate_limits['tokens']['limit'] and 
            rate_limits['tokens']['remaining'] and 
            rate_limits['tokens']['limit'] > 0):
            
            tokens_ratio = rate_limits['tokens']['remaining'] / rate_limits['tokens']['limit']
            if tokens_ratio < 0.1:  # Less than 10% remaining
                backoff_seconds = max(backoff_seconds, 60)
            elif tokens_ratio < 0.2:  # Less than 20% remaining
                backoff_seconds = max(backoff_seconds, 20)
        
        # If we're using extended thinking or output, be more conservative
        if self.use_thinking or self.use_extended_output:
            backoff_seconds = max(backoff_seconds, 10)
        
        return backoff_seconds

class TranscriptionOutput:
    """Manages transcription output files and formats"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the transcription output manager
        
        Args:
            output_dir (Path): Directory to save output files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def save_raw_transcription(self, filename: str, segments: List[Dict[str, Any]]) -> Optional[Path]:
        """
        Save raw transcription with timestamps
        
        Args:
            filename (str): Name of the file
            segments (List[Dict[str, Any]]): List of transcription segments
            
        Returns:
            Optional[Path]: Path to the output file or None if failed
        """
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
    
    def save_raw_text(self, filename: str, segments: List[Dict[str, Any]]) -> Optional[Path]:
        """
        Save raw transcription text without timestamps
        
        Args:
            filename (str): Name of the file
            segments (List[Dict[str, Any]]): List of transcription segments
            
        Returns:
            Optional[Path]: Path to the output file or None if failed
        """
        try:
            output_path = self.output_dir / f"{filename}_text.txt"
            
            with output_path.open('w', encoding='utf-8') as f:
                for segment in segments:
                    f.write(f"{segment['text']}\n")
            
            logging.info(f"Saved raw text to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving raw text: {str(e)}")
            return None
    
    def save_processed_output(self, filename: str, content: str) -> Optional[Path]:
        """
        Save processed and formatted output
        
        Args:
            filename (str): Name of the file
            content (str): Processed content to save
            
        Returns:
            Optional[Path]: Path to the output file or None if failed
        """
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
                        f.write(chunk.get('raw', 'Raw content not available'))
                        f.write("\n\n")
                        
                        f.write("PROCESSED CHUNK:\n")
                        f.write("-" * 40 + "\n")
                        f.write(chunk.get('processed', 'Processed content not available'))
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

    def create_batch_status_report(self, batch_tracker: 'BatchTracker') -> Optional[Path]:
        """
        Create batch status report showing all batches and their status
        
        Args:
            batch_tracker (BatchTracker): Batch tracker instance
            
        Returns:
            Optional[Path]: Path to the report file if successful, None otherwise
        """
        try:
            report_path = self.output_dir / f"batch_status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with report_path.open('w', encoding='utf-8') as f:
                # Write header
                f.write("=" * 80 + "\n")
                f.write(f"Batch Processing Status Report\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                # Write statistics
                stats = batch_tracker.get_batch_statistics()
                f.write("BATCH STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total batches: {stats['total']}\n")
                f.write(f"Completed batches: {stats['completed']}\n")
                f.write(f"Failed batches: {stats['failed']}\n")
                f.write(f"Pending batches: {stats['pending']}\n\n")
                
                # Write details for each batch
                f.write("BATCH DETAILS\n")
                f.write("-" * 80 + "\n\n")
                
                for filename, batch in batch_tracker.batches.items():
                    f.write(f"Filename: {filename}\n")
                    f.write(f"Status: {batch.status}\n")
                    f.write(f"Batch ID: {batch.batch_id or 'Not assigned'}\n")
                    f.write(f"Sent time: {batch.sent_time.strftime('%Y-%m-%d %H:%M:%S') if batch.sent_time else 'Not sent'}\n")
                    f.write(f"Total chunks: {len(batch.chunks)}\n")
                    
                    if batch.error_message:
                        f.write(f"Error message: {batch.error_message}\n")
                    
                    f.write(f"Retry count: {batch.retry_count}\n")
                    f.write("\n")
            
            logging.info(f"Created batch status report at {report_path}")
            return report_path
            
        except Exception as e:
            logging.error(f"Error creating batch status report: {str(e)}")
            return None
        
class SystemState:
    """Manages global system state and configuration"""
    
    def __init__(self, base_dir: Path):
        """
        Initialize the system state
        
        Args:
            base_dir (Path): Base directory for the application
        """
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
        self.batch_tracker = BatchTracker()  # Added for batch processing
        
        # State tracking
        self.active_processes: Dict[str, asyncio.Task] = {}
        self.should_stop = False
        
        # Statistics
        self.start_time = datetime.now()
        self.files_processed = 0
        self.files_failed = 0
        self.total_audio_duration = 0.0
        self.total_chunks_processed = 0
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            # Remove temporary files
            for item in self.temp_dir.glob("*"):
                if item.is_file():
                    item.unlink()
            
            # Save final progress
            self.progress_tracker.save_progress()
            
            # Log final statistics
            self.log_statistics()
            
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
    
    def log_statistics(self):
        """Log processing statistics"""
        try:
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            logging.info("=" * 40)
            logging.info("Processing Statistics")
            logging.info("=" * 40)
            logging.info(f"Total files processed: {self.files_processed}")
            logging.info(f"Total files failed: {self.files_failed}")
            logging.info(f"Total audio duration: {format_time(self.total_audio_duration)}")
            logging.info(f"Total chunks processed: {self.total_chunks_processed}")
            logging.info(f"Total processing time: {format_time(elapsed_time)}")
            
            if self.total_audio_duration > 0 and elapsed_time > 0:
                processing_ratio = self.total_audio_duration / elapsed_time
                logging.info(f"Processing ratio: {processing_ratio:.2f}x real-time")
            
            # Log batch processing statistics if available
            batch_stats = self.batch_tracker.get_batch_statistics()
            if batch_stats['total'] > 0:
                logging.info("=" * 40)
                logging.info("Batch Processing Statistics")
                logging.info("=" * 40)
                logging.info(f"Total batches: {batch_stats['total']}")
                logging.info(f"Completed batches: {batch_stats['completed']}")
                logging.info(f"Failed batches: {batch_stats['failed']}")
                logging.info(f"Pending batches: {batch_stats['pending']}")
            
            logging.info("=" * 40)
            
        except Exception as e:
            logging.error(f"Error logging statistics: {str(e)}")
    
    def update_statistics(self, file_duration: float, chunks_processed: int, success: bool):
        """
        Update processing statistics
        
        Args:
            file_duration (float): Duration of the processed file in seconds
            chunks_processed (int): Number of chunks processed
            success (bool): Whether processing was successful
        """
        if success:
            self.files_processed += 1
            self.total_audio_duration += file_duration
            self.total_chunks_processed += chunks_processed
        else:
            self.files_failed += 1

class FileProcessor:
    """Handles file processing workflow and state management"""
    
    def __init__(self, system_state: SystemState, model: EnhancedWhisperModel, text_processor: TextProcessor):
        """
        Initialize the file processor
        
        Args:
            system_state (SystemState): System state instance
            model (EnhancedWhisperModel): Enhanced Whisper model instance
            text_processor (TextProcessor): Text processor instance
        """
        self.system_state = system_state
        self.model = model
        self.text_processor = text_processor
        self.max_retries = 3
        self.retry_delay = 30
        self.chunk_overlap = 0.5
    
    async def prepare_file(self, file_path: Path) -> Optional[Path]:
        """
        Prepare file for processing with enhanced validation
        
        Args:
            file_path (Path): Path to the file
            
        Returns:
            Optional[Path]: Path to the prepared file or None if failed
        """
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
        Process a single file with progress tracking and error handling,
        supporting both regular and batch processing modes.
        
        Args:
            file_path (Path): Path to the file to process.
        
        Returns:
            bool: True if processing is successful or batch submitted, False otherwise.
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
            
            # Always save raw transcriptions immediately, regardless of processing mode
            raw_path = self.system_state.output_manager.save_raw_transcription(
                original_filename,
                segments
            )
            
            text_path = self.system_state.output_manager.save_raw_text(
                original_filename,
                segments
            )
            
            # Combine segments into text
            full_text = "\n".join(segment['text'] for segment in segments)
            
            # Update progress for text processing stage
            progress.current_action = "Processing transcription"
            self.system_state.progress_tracker.update_progress(
                original_filename,
                current_action=progress.current_action
            )
            
            # Process text based on mode (regular or batch)
            if self.text_processor.use_batch_api:
                # Batch API processing path
                
                # Split text into chunks
                chunks = self.text_processor.split_into_chunks(full_text)
                if not chunks:
                    logging.error(f"Failed to split text into chunks for {original_filename}")
                    return False
                
                # Initialize batch in batch tracker
                batch_info = self.system_state.batch_tracker.initialize_batch(original_filename, chunks)
                
                # Submit for batch processing - this doesn't wait for results
                batch_id = await self.text_processor._process_text_batch(chunks, progress)

                if not batch_id:
                    logging.error(f"Failed to submit batch for {original_filename}")
                    return False

                # וודא שה-batch_id הוא אכן מחרוזת תקינה
                if not isinstance(batch_id, str) or not batch_id.startswith("msgbatch_"):
                    logging.error(f"Invalid batch_id received: {batch_id}")
                    return False

                # Update batch tracking with batch ID
                self.system_state.batch_tracker.update_batch_submission(original_filename, batch_id)

                # We don't wait for batch results, mark as pending
                progress.current_action = f"Batch submitted (ID: {batch_id})"
                self.system_state.progress_tracker.update_progress(
                    original_filename,
                    current_action=progress.current_action
                )

                # Update statistics for this file
                self.system_state.update_statistics(
                    progress.total_duration,
                    len(chunks),
                    True  # Consider batch submission a success
                )
                
                # Clean up temporary files
                if prepared_path != file_path:
                    prepared_path.unlink()
                
                # We return True here even though processing isn't complete,
                # as batch processing will be completed later
                return True
                
            else:
                # Regular API processing path
                processed_text = await self.text_processor.process_text(full_text, progress)
                
                if not processed_text:
                    return False
                
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
                
                # Update statistics
                self.system_state.update_statistics(
                    progress.total_duration,
                    progress.total_chunks,
                    True
                )
                
                # Clean up
                if prepared_path != file_path:
                    prepared_path.unlink()
                
                return True
            
        except Exception as e:
            logging.error(f"Error processing file {original_filename}: {str(e)}")
            self.system_state.update_statistics(0, 0, False)
            return False
        
class ProcessManager:
    """Manages concurrent file processing and resource allocation, with additional
    support for batch processing and collecting results from pending batches."""
    
    def __init__(self, system_state: SystemState, max_concurrent: int = 1):
        """
        Initialize the process manager
        
        Args:
            system_state (SystemState): System state instance
            max_concurrent (int, optional): Maximum number of concurrent processes. Defaults to 1.
        """
        self.system_state = system_state
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.batch_poll_interval = 30  # seconds
    
    async def process_file_with_semaphore(self, processor: FileProcessor, file_path: Path):
        """
        Process a file with resource limiting
        
        Args:
            processor (FileProcessor): File processor instance
            file_path (Path): Path to the file to process
        """
        async with self.semaphore:
            try:
                await processor.process_file(file_path)
            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
    
    async def process_directory(self, processor: FileProcessor, directory: Path):
        """
        Process all files in directory with concurrency control
        
        Args:
            processor (FileProcessor): File processor instance
            directory (Path): Directory containing files to process
        """
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
                
                # If batch API was used, process any pending batches after file transcription
                if processor.text_processor.use_batch_api:
                    await self.process_pending_batches(processor)
            
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
    
    async def check_batch_status(self, processor: FileProcessor, batch_info: BatchInfo) -> str:
        """
        Check the status of a batch
        
        Args:
            processor (FileProcessor): File processor instance
            batch_info (BatchInfo): Batch information
            
        Returns:
            str: Current batch status
        """
        try:
            # Skip if batch has no ID or is already completed or failed
            if not batch_info.batch_id or batch_info.status in ["completed", "error"]:
                return batch_info.status
            
            # Check current status
            status = await processor.text_processor.check_batch_status(batch_info.batch_id)
            
            if not status:
                # If status check fails, increment retry count
                if batch_info.retry_count >= processor.max_retries:
                    self.system_state.batch_tracker.update_batch_status(
                        batch_info.filename,
                        "error",
                        "Max retries reached checking batch status"
                    )
                    return "error"
                else:
                    self.system_state.batch_tracker.increment_retry_count(batch_info.filename)
                    return batch_info.status  # Keep current status
            
            # Update status in tracker
            self.system_state.batch_tracker.update_batch_status(batch_info.filename, status)
            return status
            
        except Exception as e:
            logging.error(f"Error checking batch status for {batch_info.filename}: {str(e)}")
            return batch_info.status  # Keep current status
    
    def process_batch_results(self, batch_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Process batch results to extract improved text for each chunk
        
        Args:
            batch_results (Dict[str, Any]): Batch results from API
            
        Returns:
            Dict[str, str]: Dictionary mapping custom IDs to processed text
        """
        processed_results = {}
        
        try:
            # Process each output in the batch
            outputs = batch_results.get('outputs', [])
            
            # Log the structure of the first output for debugging if available
            if outputs and len(outputs) > 0:
                logging.debug(f"First output structure: {list(outputs[0].keys() if outputs[0] else {})}")
            
            for output in outputs:
                custom_id = output.get('custom_id')
                
                # Check if output has error
                error = output.get('error')
                if error:
                    logging.warning(f"Output {custom_id} has error: {error}")
                    continue
                    
                if not custom_id:
                    logging.warning("Output missing custom_id")
                    continue
                
                # Extract message content - structure matches regular API response
                message = output.get('result', {}).get('message', {})
                content = message.get('content', [])
                
                if not content:
                    logging.warning(f"Output {custom_id} missing content")
                    continue
                
                # Extract text content from the first content block
                text_content = ""
                for block in content:
                    if block.get('type') == 'text':
                        text_content += block.get('text', '')
                
                if not text_content:
                    logging.warning(f"Output {custom_id} has no text content")
                    continue
                
                # Extract text between improved_text tags
                match = re.search(r'<improved_text>\s*(.*?)(?=</improved_text>|$)', text_content, re.DOTALL | re.IGNORECASE)
                if not match:
                    logging.warning(f"Output {custom_id} missing improved_text tags")
                    continue
                
                improved_text = match.group(1).strip()
                if not improved_text:
                    logging.warning(f"Output {custom_id} has empty improved text")
                    continue
                
                processed_results[custom_id] = improved_text
            
            logging.info(f"Processed {len(processed_results)} results from batch")
            return processed_results
            
        except Exception as e:
            logging.error(f"Error processing batch results: {str(e)}")
            logging.error(traceback.format_exc())
            return processed_results
        
    async def process_pending_batches(self, processor: FileProcessor):
        """
        Process all pending batches - check status, retrieve results when complete
        
        Args:
            processor (FileProcessor): File processor instance
        """
        try:
            # Get pending batches
            pending_batches = self.system_state.batch_tracker.get_pending_batches()
            
            if not pending_batches:
                logging.info("No pending batches to process")
                return
            
            # Set up progress tracking
            batch_progress = self.system_state.progress_tracker.start_batch_tracking(len(pending_batches))
            logging.info(f"Processing {len(pending_batches)} pending batches")
            
            processed_count = 0
            max_poll_attempts = 100  # גדל מ-10 ל-100 לאפשר יותר בדיקות סטטוס
            poll_interval_seconds = self.batch_poll_interval
            
            # Process each batch with proper error handling
            for batch_info in pending_batches[:]:  # Work on a copy to safely modify the original list
                poll_attempt = 0
                filename = batch_info.filename
                
                # Skip batch if batch_id is invalid
                if not batch_info.batch_id or not isinstance(batch_info.batch_id, str) or not batch_info.batch_id.startswith("msgbatch_"):
                    logging.error(f"Invalid batch_id: '{batch_info.batch_id}' for file '{filename}'")
                    self.system_state.batch_tracker.update_batch_status(
                        filename, 
                        "error", 
                        f"Invalid batch_id format: {str(batch_info.batch_id)[:50]}..."
                    )
                    processed_count += 1
                    self.system_state.progress_tracker.update_batch_progress(processed_count)
                    continue
                
                # Poll batch status with limited attempts
                while poll_attempt < max_poll_attempts and not self.system_state.should_stop:
                    batch_id = batch_info.batch_id
                    status_info = await processor.text_processor.check_batch_status(batch_id)
                    
                    # Handle failed status check
                    if not status_info:
                        poll_attempt += 1
                        logging.warning(f"Failed to get status for batch {batch_id}, attempt {poll_attempt}/{max_poll_attempts}")
                        if poll_attempt >= max_poll_attempts:
                            self.system_state.batch_tracker.update_batch_status(
                                filename,
                                "error",
                                "Max retries reached checking batch status"
                            )
                            processed_count += 1
                            self.system_state.progress_tracker.update_batch_progress(processed_count)
                            break
                        
                        # Exponential backoff
                        backoff_time = min(poll_interval_seconds * (2 ** (poll_attempt - 1)), 300)  # max 5 minutes
                        logging.info(f"Backing off for {backoff_time} seconds before next check")
                        await asyncio.sleep(backoff_time)
                        continue
                    
                    status = status_info["status"]
                    request_counts = status_info.get("request_counts", {})
                    
                    # Update progress based on request counts
                    if request_counts and "succeeded" in request_counts and "processing" in request_counts:
                        total = sum(request_counts.values())
                        completed = request_counts.get("succeeded", 0) + request_counts.get("errored", 0) + request_counts.get("expired", 0)
                        if total > 0:
                            percentage = (completed / total) * 100
                            logging.info(f"Batch {batch_id} progress: {percentage:.1f}% ({completed}/{total})")
                    
                    # Handle rate limiting
                    if status == "rate_limited":
                        retry_after = status_info.get("retry_after", poll_interval_seconds)
                        logging.warning(f"Rate limited. Waiting {retry_after} seconds before retrying")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    # Update batch status in tracker
                    self.system_state.batch_tracker.update_batch_status(filename, status)
                    
                    # Process completed batch
                    if status == "ended":
                        logging.info(f"Batch {batch_id} for file '{filename}' has ended. Retrieving results...")
                        
                        # Retrieve results
                        results = await processor.text_processor.retrieve_batch_results(batch_id)
                        
                        if results and "results" in results:
                            # Process results successfully
                            logging.info(f"Successfully retrieved {len(results['results'])} results for batch {batch_id}")
                            
                            # Update individual chunks with their processed content
                            for custom_id, processed_content in results["results"].items():
                                # Try to map sanitized ID back to original if needed
                                if hasattr(processor.text_processor, 'custom_id_mapping') and custom_id in processor.text_processor.custom_id_mapping:
                                    original_id = processor.text_processor.custom_id_mapping.get(custom_id)
                                    self.system_state.batch_tracker.update_chunk_result(
                                        filename, 
                                        original_id, 
                                        processed_content,
                                        results.get("thinking_blocks", {}).get(custom_id)
                                    )
                                else:
                                    self.system_state.batch_tracker.update_chunk_result(
                                        filename, 
                                        custom_id, 
                                        processed_content,
                                        results.get("thinking_blocks", {}).get(custom_id)
                                    )
                            
                            # Get all processed chunks in order
                            ordered_results = self.system_state.batch_tracker.get_all_chunk_results(filename)
                            
                            if ordered_results:
                                # Combine processed text and save it
                                processed_parts = [content for _, content in ordered_results]
                                combined_text = "\n\n".join(processed_parts)
                                
                                # Save the processed output
                                processor_path = self.system_state.output_manager.save_processed_output(
                                    filename,
                                    combined_text
                                )
                                
                                if processor_path:
                                    logging.info(f"Saved processed output to {processor_path}")
                                    
                                    # Mark batch as completed
                                    self.system_state.batch_tracker.update_batch_status(filename, "completed")
                                    self.system_state.progress_tracker.mark_completed(filename)
                                else:
                                    logging.error(f"Failed to save processed output for {filename}")
                                    self.system_state.batch_tracker.update_batch_status(filename, "error", "Failed to save processed output")
                            else:
                                logging.error(f"No processed content for any chunks in {filename}")
                                self.system_state.batch_tracker.update_batch_status(filename, "error", "No processed content for chunks")
                        else:
                            logging.error(f"Failed to retrieve results for batch {batch_id}")
                            self.system_state.batch_tracker.update_batch_status(filename, "error", "Failed to retrieve batch results")
                        
                        processed_count += 1
                        self.system_state.progress_tracker.update_batch_progress(processed_count)
                        break
                        
                    elif status == "in_progress":
                        # If still in progress, wait and check again
                        poll_attempt += 1
                        
                        # Use adaptive backoff based on progress and total requests
                        progress_percent = 0
                        if request_counts:
                            total = sum(request_counts.values())
                            completed = request_counts.get("succeeded", 0) + request_counts.get("errored", 0)
                            if total > 0:
                                progress_percent = (completed / total) * 100
                        
                        # Adjust polling interval based on progress
                        if progress_percent > 80:
                            # Almost done, check more frequently
                            adjusted_interval = poll_interval_seconds / 2
                        elif progress_percent > 50:
                            adjusted_interval = poll_interval_seconds
                        else:
                            # Still early, check less frequently
                            adjusted_interval = poll_interval_seconds * 2
                            
                        logging.info(f"Batch {batch_id} is still in progress ({progress_percent:.1f}%). "
                                    f"Poll attempt {poll_attempt}/{max_poll_attempts}. "
                                    f"Checking again in {adjusted_interval} seconds.")
                        
                        await asyncio.sleep(adjusted_interval)
                    else:
                        # Handle other statuses (error, canceled, etc.)
                        logging.warning(f"Batch {batch_id} has status: {status}")
                        if status == "error":
                            self.system_state.batch_tracker.update_batch_status(filename, status, f"Batch ended with error status")
                        else:
                            self.system_state.batch_tracker.update_batch_status(filename, status)
                        processed_count += 1
                        self.system_state.progress_tracker.update_batch_progress(processed_count)
                        break
                
                # Handle case where max poll attempts reached
                if poll_attempt >= max_poll_attempts and not self.system_state.should_stop:
                    logging.warning(f"Reached maximum poll attempts for batch {batch_info.batch_id}")
                    self.system_state.batch_tracker.update_batch_status(
                        filename,
                        "error",
                        "Max poll attempts reached"
                    )
                    processed_count += 1
                    self.system_state.progress_tracker.update_batch_progress(processed_count)
            
            # Create batch status report
            self.system_state.output_manager.create_batch_status_report(self.system_state.batch_tracker)
            logging.info("Batch processing completed")
            
        except Exception as e:
            logging.error(f"Error processing pending batches: {str(e)}")
            logging.error(traceback.format_exc())

@dataclass
class ApplicationConfig:
    """Application configuration with validation and support for Batch API, Extended Thinking and Output"""
    
    # Path configuration
    input_dir: str = "input"
    output_dir: str = "output"
    temp_dir: str = "temp"
    checkpoint_dir: str = "checkpoints"
    
    # API configuration
    claude_api_key: str = ""
    use_batch_api: bool = True  # Changed default to True based on official docs recommendation
    
    # Extended thinking and output
    use_thinking: bool = True  # Changed default to True for better quality results
    thinking_budget_tokens: int = 16000  # Increased from 8192 based on documentation
    use_extended_output: bool = True  # Keep as False by default, but offer to user for large files
    
    # Model configuration
    model_name: str = "ivrit-ai/whisper-large-v3-turbo-ct2"
    compute_type: str = "float16"
    
    # Processing configuration  
    max_concurrent_files: int = 1
    chunk_size: int = 3000  # Default for regular processing
    overlap_size: int = 200  # Increased from 100 for better continuity
    max_retries: int = 3
    retry_delay: int = 30
    
    # Language configuration
    source_language: str = "he"  # Hebrew
    target_language: str = "he"  # Hebrew
    
    # Claude configuration
    claude_model: str = "claude-3-7-sonnet-20250219"
    claude_max_tokens: int = 8192
    
    # Batch API configuration
    batch_poll_interval: int = 15  # Reduced from 30 to check more frequently at the start
    
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
        if self.chunk_size < 1500:
            raise ValueError("chunk_size must be at least 1500")
        if self.overlap_size >= self.chunk_size:
            raise ValueError("overlap_size must be less than chunk_size")
        if self.max_retries < 1:  
            raise ValueError("max_retries must be at least 1")
        if self.retry_delay < 1:
            raise ValueError("retry_delay must be at least 1")
        if self.batch_poll_interval < 5:
            raise ValueError("batch_poll_interval must be at least 5 seconds")
        
        # Extended thinking validation
        if self.use_thinking and self.thinking_budget_tokens < 1024:
            raise ValueError("thinking_budget_tokens must be at least 1024")
        
        # Set up extended output parameters
        if self.use_extended_output:
            # If using extended output, adjust max_tokens accordingly
            self.claude_max_tokens = 128000
            
            # Optimize chunk size for extended output
            if self.chunk_size < 20000:
                orig_chunk_size = self.chunk_size
                self.chunk_size = 25000
                logging.info(f"Adjusted chunk_size from {orig_chunk_size} to {self.chunk_size} for extended output mode")
            
            # Increase overlap for better continuity with larger chunks
            if self.overlap_size < 200:
                orig_overlap = self.overlap_size
                self.overlap_size = max(500, self.chunk_size // 50)
                logging.info(f"Adjusted overlap_size from {orig_overlap} to {self.overlap_size} for extended output mode")
                
        # Optimize thinking budget based on extended output setting
        if self.use_extended_output and self.use_thinking:
            # Save original value for logging
            original_budget = self.thinking_budget_tokens
            
            if self.chunk_size >= 15000:
                # For very large chunks, can increase budget substantially (not exceeding safe limits)
                self.thinking_budget_tokens = min(32000, original_budget * 2)
            elif self.chunk_size >= 8000:
                # For medium chunks, moderate increase
                self.thinking_budget_tokens = min(24000, original_budget * 1.5)
            else:
                # For smaller chunks, small increase
                self.thinking_budget_tokens = min(16000, original_budget * 1.2)
            
            # Log adjustment if made
            if self.thinking_budget_tokens != original_budget:
                logging.info(f"Adjusted thinking budget from {original_budget} to {self.thinking_budget_tokens} tokens based on extended output and chunk size")
            
            # Always ensure thinking budget doesn't exceed recommended limits for network stability
            if self.thinking_budget_tokens > 32000:
                logging.warning("Reducing thinking_budget_tokens to 32000 to avoid network issues")
                self.thinking_budget_tokens = 32000
                
        # Always ensure thinking budget doesn't consume too much of the output capacity in extended output mode
        if self.use_extended_output and self.thinking_budget_tokens > 48000:
            self.thinking_budget_tokens = 48000
            logging.info(f"Capping thinking budget to {self.thinking_budget_tokens} tokens to reserve capacity for output")
            
        # Set default input_dir if not provided or empty - הקוד הקיים שלך ממשיך כאן
        if not self.input_dir:
            script_dir = Path(__file__).parent  # Get script directory
            self.input_dir = str(script_dir / "input")  # Create path to input directory
            # Create input directory if it doesn't exist
            Path(self.input_dir).mkdir(exist_ok=True)
            logging.info(f"Using default input directory: {self.input_dir}")

class TranscriptionSystem:
    """Main transcription system orchestrator with Batch API support"""
    
    def __init__(self, config: ApplicationConfig):
        """
        Initialize the transcription system
        
        Args:
            config (ApplicationConfig): Application configuration
        """
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
        """Initialize all system components with extended capabilities"""
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
                    self.system_state.progress_tracker,
                    self.config.use_batch_api,  # Use batch API if configured
                    self.config.use_thinking,   # Use extended thinking if configured
                    self.config.thinking_budget_tokens,  # Budget for extended thinking
                    self.config.use_extended_output  # Use extended output if configured
                )
                    
                # Update batch API parameters
                if self.config.use_batch_api:
                    TextProcessor.CHUNK_MAX_SIZE = self.config.chunk_size
                    TextProcessor.CHUNK_MIN_SIZE = self.config.chunk_size // 10
                    TextProcessor.OVERLAP_SIZE = self.config.overlap_size
                    
                # Log initialization details
                features = []
                if self.config.use_batch_api:
                    features.append("Batch API")
                if self.config.use_thinking:
                    features.append(f"Extended Thinking ({self.config.thinking_budget_tokens} tokens)")
                if self.config.use_extended_output:
                    features.append(f"Extended Output (128K tokens)")
                    
                logging.info(f"Text processor initialized with: {', '.join(features) or 'Standard API'} mode")
            else:
                logging.info("No Claude API key provided. Skipping text processing.")
            
            # Initialize file processor
            self.file_processor = FileProcessor(
                self.system_state,
                self.model,
                self.text_processor
            )
            
            # Update file processor parameters
            self.file_processor.max_retries = self.config.max_retries
            self.file_processor.retry_delay = self.config.retry_delay
            
            # Initialize process manager  
            self.process_manager = ProcessManager(
                self.system_state,
                self.config.max_concurrent_files
            )
            
            # Update process manager parameters
            self.process_manager.batch_poll_interval = self.config.batch_poll_interval
            
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
            batch_mode = "Batch API" if self.config.use_batch_api else "Regular API"
            console.print(Panel(
                f"[bold green]Transcription System Starting[/bold green]\n\n"
                f"Input Directory: {input_dir}\n"
                f"Output Directory: {self.system_state.output_dir}\n"
                f"Model: {self.config.model_name}\n"
                f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n"
                f"API Mode: {batch_mode}"
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

async def check_existing_batches():
    """Check and process existing pending batches from previous runs"""
    try:
        # Initialize configuration
        config = ApplicationConfig(
            # Minimal configuration needed
            claude_api_key=input("Enter the Claude API key: ").strip(),
            use_batch_api=True
        )
        
        # Create and initialize system
        system = TranscriptionSystem(config)
        if not system.initialize_components():
            console.print("[bold red]Failed to initialize system components.[/bold red]")
            return False
        
        # Process pending batches
        pending_batches = system.system_state.batch_tracker.get_pending_batches()
        if not pending_batches:
            console.print("[bold yellow]No pending batches found.[/bold yellow]")
            return True
        
        console.print(f"[bold green]Found {len(pending_batches)} pending batches to process.[/bold green]")
        
        # Process pending batches
        await system.process_manager.process_pending_batches(system.file_processor)
        
        console.print("[bold green]Batch processing completed successfully![/bold green]")
        return True
        
    except Exception as e:
        logging.error(f"Critical error processing pending batches: {str(e)}")
        logging.error(traceback.format_exc())
        console.print(f"[bold red]Critical error: {str(e)}[/bold red]")
        return False

async def main():
    """Main entry point for the transcription system with extended capabilities"""
    try:
        # Get input directory from user
        input_dir = input("Enter the input directory path (leave blank to use default 'input' directory): ").strip()
        # If input_dir is empty, it will use the default from ApplicationConfig

        # Get Claude API key from user (optional)
        api_key = input("Enter the Claude API key (leave blank to skip text processing): ").strip()
        
        use_batch_api = True  # Changed default based on recommendations
        use_thinking = True   # Changed default based on recommendations
        thinking_budget_tokens = 16000  # Higher default based on recommendations
        use_extended_output = False
        chunk_size = 3000  # Default chunk size for regular processing
        
        if api_key:
            # Ask about Batch API usage - now default is YES
            batch_choice = input("Do you want to use Batch API for more efficient processing (50% cost savings)? (Y/n): ").strip().lower()
            use_batch_api = batch_choice not in ('n', 'no')  # Default is True!
            
            if use_batch_api:
                print("Using Batch API mode for more efficient processing (50% cost savings).")
            else:
                print("Using Regular API mode with sequential processing.")
            
            # Ask about Extended Thinking - now default is YES
            thinking_choice = input("Do you want to use Extended Thinking for better results? (Y/n): ").strip().lower()
            use_thinking = thinking_choice not in ('n', 'no')  # Default is True!
            
            if use_thinking:
                budget_choice = input("Enter thinking budget in tokens (1024-32000, leave blank for recommended 16000): ").strip()
                if budget_choice and budget_choice.isdigit():
                    thinking_budget = int(budget_choice)
                    if 1024 <= thinking_budget <= 32000:
                        thinking_budget_tokens = thinking_budget
                        print(f"Using extended thinking with {thinking_budget_tokens} token budget")
                    else:
                        print(f"Invalid budget, using recommended {thinking_budget_tokens} tokens")
                else:
                    print(f"Using recommended thinking budget of {thinking_budget_tokens} tokens")
            
            # Check for large files to suggest Extended Output
            files = []
            input_path = Path(input_dir if input_dir else "input")
            if input_path.exists():
                for ext in FileValidator.SUPPORTED_EXTENSIONS:
                    files.extend(input_path.glob(f"*{ext}"))
                
                # Check for large files (over 20MB or long duration)
                large_files = [file for file in files if file.stat().st_size > 1]
                if large_files:
                    print(f"\nDetected {len(large_files)} large files that may benefit from Extended Output capability:")
                    for file in large_files[:3]:  # Show first 3 as examples
                        print(f"  - {file.name} ({file.stat().st_size // (1024*1024)} MB)")
                    if len(large_files) > 3:
                        print(f"  - ...and {len(large_files) - 3} more")
                        
                    # Suggest Extended Output - stronger recommendation for large files
                    output_choice = input("Do you want to use Extended Output for these large files (up to 128K tokens)? (Y/n): ").strip().lower()
                    use_extended_output = output_choice not in ('n', 'no')  # Default is True for large files
                else:
                    # Still offer but with softer recommendation
                    output_choice = input("Do you want to use Extended Output capability (up to 128K tokens)? (y/N): ").strip().lower()
                    use_extended_output = output_choice in ('y', 'yes')  # Default is False
            else:
                # Ask without looking at files
                output_choice = input("Do you want to use Extended Output capability (up to 128K tokens)? (y/N): ").strip().lower()
                use_extended_output = output_choice in ('y', 'yes')
            
            if use_extended_output:
                print("Using Extended Output capability (up to 128K tokens)")
                
                # Suggest optimal chunk size
                chunk_size_input = input(f"Enter maximum chunk size (recommended 25000 for extended output, leave blank for default): ").strip()
                if chunk_size_input and chunk_size_input.isdigit() and int(chunk_size_input) > 1500:
                    chunk_size = int(chunk_size_input)
                else:
                    chunk_size = 25000  # Default for extended output
                    
                print(f"Using chunk size of {chunk_size} tokens with Extended Output")
        
        # Load configuration with updated defaults
        model_name = input("Enter Whisper model name (leave blank for default 'ivrit-ai/whisper-large-v3-turbo-ct2'): ").strip()
        
        config = ApplicationConfig(
            input_dir=input_dir if input_dir else "",
            claude_api_key=api_key,
            use_batch_api=use_batch_api,
            use_thinking=use_thinking,
            thinking_budget_tokens=thinking_budget_tokens,
            use_extended_output=use_extended_output,
            model_name=model_name if model_name else "ivrit-ai/whisper-large-v3-turbo-ct2",
            chunk_size=chunk_size
        )
        
        # Continue with the rest of the function as before...
        
        # Prompt user for customizations
        if api_key:
            # Claude model
            model_choice = input("Enter Claude model name (leave blank for default 'claude-3-7-sonnet-20250219'): ").strip()
            if model_choice:
                config.claude_model = model_choice
            
            # Concurrent processing
            concurrent_choice = input("Enter maximum concurrent files to process (leave blank for default '1'): ").strip()
            if concurrent_choice and concurrent_choice.isdigit() and int(concurrent_choice) > 0:
                config.max_concurrent_files = int(concurrent_choice)
            
            # Batch poll interval (only relevant for batch API)
            if use_batch_api:
                poll_choice = input("Enter batch status poll interval in seconds (leave blank for default '30'): ").strip()
                if poll_choice and poll_choice.isdigit() and int(poll_choice) >= 5:
                    config.batch_poll_interval = int(poll_choice)
            
            # Prompt user for custom Claude prompt (optional)
            print("\nDefault Claude prompt:")
            print(TextProcessor.CLAUDE_PROMPT)
            choice = input("Enter 'v' to use the default prompt or 'new' to provide a custom one: ")
            if choice.lower() == 'new':
                print("Enter the custom Claude prompt (type 'END' on a new line when finished):")
                lines = []
                while True:
                    line = input()
                    if line == 'END':
                        break
                    lines.append(line)
                custom_prompt = '\n'.join(lines)
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
        logging.error(traceback.format_exc())
        console.print(f"[bold red]Critical error: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    # Set up asyncio event loop with error handling
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Ask user if they want to process new files or check existing batches
        mode = input("Choose mode - (1) Process new files or (2) Check existing batches: ").strip()
        
        if mode == "2":
            asyncio.run(check_existing_batches())
        else:
            asyncio.run(main())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        sys.exit(0)  
    except Exception as e:
        console.print(f"[bold red]Fatal error: {str(e)}[/bold red]")
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)    # Set up asyncio event loop with error handling
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(main())
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        sys.exit(0)  
    except Exception as e:
        console.print(f"[bold red]Fatal error: {str(e)}[/bold red]")
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
