from nemo.collections.asr.models import ASRModel
import torch
import gc
import shutil
from pathlib import Path
from pydub import AudioSegment
import numpy as np
import datetime
import tempfile
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

# Global model instance
model = None

def load_model():
    """Load the ASR model on startup"""
    global model
    try:
        model = ASRModel.from_pretrained(model_name=MODEL_NAME)
        model.eval()
        print(f"Model {MODEL_NAME} loaded successfully on {device}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None

# Pydantic models for API
class TranscriptionSegment(BaseModel):
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds") 
    text: str = Field(..., description="Transcribed text segment")

class TranscriptionResponse(BaseModel):
    success: bool = Field(..., description="Whether transcription was successful")
    segments: List[TranscriptionSegment] = Field(default=[], description="List of transcription segments")
    duration: Optional[float] = Field(None, description="Total audio duration in seconds")
    message: Optional[str] = Field(None, description="Status message or error description")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether the ASR model is loaded")
    device: str = Field(..., description="Device being used (cuda/cpu)")

# Initialize FastAPI app
app = FastAPI(
    title="Speech Transcription API",
    description="A REST API for speech-to-text transcription using NVIDIA's parakeet-tdt-0.6b-v2 model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

def cleanup_session_dir(session_dir: Path):
    """Background task to clean up session directory"""
    try:
        if session_dir.exists():
            shutil.rmtree(session_dir)
            print(f"Cleaned up session directory: {session_dir}")
    except Exception as e:
        print(f"Error cleaning up session directory {session_dir}: {e}")

def get_audio_segment(audio_path: str, start_second: float, end_second: float):
    """Extract audio segment from file between start and end times"""
    if not audio_path or not Path(audio_path).exists():
        print(f"Warning: Audio path '{audio_path}' not found or invalid for clipping.")
        return None
    try:
        start_ms = int(start_second * 1000)
        end_ms = int(end_second * 1000)

        start_ms = max(0, start_ms)
        if end_ms <= start_ms:
            print(f"Warning: End time ({end_second}s) is not after start time ({start_second}s). Adjusting end time.")
            end_ms = start_ms + 100

        audio = AudioSegment.from_file(audio_path)
        clipped_audio = audio[start_ms:end_ms]

        samples = np.array(clipped_audio.get_array_of_samples())
        if clipped_audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1).astype(samples.dtype)

        frame_rate = clipped_audio.frame_rate
        if frame_rate <= 0:
             print(f"Warning: Invalid frame rate ({frame_rate}) detected for clipped audio.")
             frame_rate = audio.frame_rate

        if samples.size == 0:
             print(f"Warning: Clipped audio resulted in empty samples array ({start_second}s to {end_second}s).")
             return None

        return (frame_rate, samples)
    except FileNotFoundError:
        print(f"Error: Audio file not found at path: {audio_path}")
        return None
    except Exception as e:
        print(f"Error clipping audio {audio_path} from {start_second}s to {end_second}s: {e}")
        return None

def format_srt_time(seconds: float) -> str:
    """Converts seconds to SRT time format HH:MM:SS,mmm using datetime.timedelta"""
    sanitized_total_seconds = max(0.0, seconds)
    delta = datetime.timedelta(seconds=sanitized_total_seconds)
    total_int_seconds = int(delta.total_seconds())

    hours = total_int_seconds // 3600
    remainder_seconds_after_hours = total_int_seconds % 3600
    minutes = remainder_seconds_after_hours // 60
    seconds_part = remainder_seconds_after_hours % 60
    milliseconds = delta.microseconds // 1000

    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"

def generate_srt_content(segment_timestamps: List[Dict]) -> str:
    """Generates SRT formatted string from segment timestamps."""
    srt_content = []
    for i, ts in enumerate(segment_timestamps):
        start_time = format_srt_time(ts['start'])
        end_time = format_srt_time(ts['end'])
        text = ts['segment']
        srt_content.append(str(i + 1))
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(text)
        srt_content.append("")
    return "\n".join(srt_content)

def process_audio_for_transcription(audio_path: str, session_dir: Path) -> tuple:
    """Process audio file for transcription (resampling, mono conversion)"""
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_sec = audio.duration_seconds
        
        resampled = False
        mono = False
        
        # Resample to 16kHz if needed
        target_sr = 16000
        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)
            resampled = True
            
        # Convert to mono if needed
        if audio.channels == 2:
            audio = audio.set_channels(1)
            mono = True
        elif audio.channels > 2:
            raise ValueError(f"Audio has {audio.channels} channels. Only mono (1) or stereo (2) supported.")
            
        # Export processed audio if changes were made
        if resampled or mono:
            audio_name = Path(audio_path).stem
            processed_audio_path = session_dir / f"{audio_name}_processed.wav"
            audio.export(processed_audio_path, format="wav")
            return processed_audio_path.as_posix(), duration_sec
        else:
            return audio_path, duration_sec
            
    except Exception as e:
        raise RuntimeError(f"Failed to process audio: {e}")

def get_transcripts_and_raw_times(audio_path: str, session_dir: Path) -> TranscriptionResponse:
    """Main transcription function"""
    if not model:
        return TranscriptionResponse(
            success=False,
            message="ASR model is not loaded"
        )
    
    if not audio_path:
        return TranscriptionResponse(
            success=False,
            message="No audio file path provided"
        )
    
    try:
        # Process audio
        transcribe_path, duration_sec = process_audio_for_transcription(audio_path, session_dir)
        
        # Configure model for long audio if needed
        long_audio_settings_applied = False
        try:
            model.to(device)
            model.to(torch.float32)
            
            # Apply settings for long audio (>8 minutes)
            if duration_sec > 480:
                print("Applying long audio settings: Local Attention and Chunking.")
                model.change_attention_model("rel_pos_local_attn", [256, 256])
                model.change_subsampling_conv_chunking_factor(1)
                long_audio_settings_applied = True
            
            # Perform transcription
            model.to(torch.bfloat16)
            output = model.transcribe([transcribe_path], timestamps=True)
            
            if not output or not isinstance(output, list) or not output[0] or \
               not hasattr(output[0], 'timestamp') or not output[0].timestamp or \
               'segment' not in output[0].timestamp:
                return TranscriptionResponse(
                    success=False,
                    message="Transcription failed or produced unexpected output format"
                )
            
            segment_timestamps = output[0].timestamp['segment']
            
            # Convert to response format
            segments = [
                TranscriptionSegment(
                    start=ts['start'],
                    end=ts['end'],
                    text=ts['segment']
                )
                for ts in segment_timestamps
            ]
            
            return TranscriptionResponse(
                success=True,
                segments=segments,
                duration=duration_sec,
                message="Transcription completed successfully"
            )
            
        finally:
            # Revert model settings if applied
            if long_audio_settings_applied:
                try:
                    print("Reverting long audio settings.")
                    model.change_attention_model("rel_pos")
                    model.change_subsampling_conv_chunking_factor(-1)
                except Exception as e:
                    print(f"Warning: Failed to revert long audio settings: {e}")
            
            # Cleanup
            try:
                if device == 'cuda':
                    model.cpu()
                gc.collect()
                if device == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error during model cleanup: {e}")
                
    except torch.cuda.OutOfMemoryError as e:
        return TranscriptionResponse(
            success=False,
            message="CUDA out of memory. Please try a shorter audio or reduce GPU load."
        )
    except Exception as e:
        return TranscriptionResponse(
            success=False,
            message=f"Transcription failed: {str(e)}"
        )

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model else "unhealthy",
        model_loaded=model is not None,
        device=device
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to transcribe")
):
    """
    Transcribe an audio file to text with timestamps
    
    - **file**: Audio file (supported formats: wav, mp3, flac, etc.)
    
    Returns transcription segments with start/end timestamps and text
    """
    # Validate file type
    allowed_types = ["audio/wav", "audio/mpeg", "audio/flac", "audio/ogg", "audio/mp4"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: {allowed_types}"
        )
    
    # Create session directory
    session_id = str(uuid.uuid4())
    session_dir = Path(tempfile.gettempdir()) / f"transcription_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_session_dir, session_dir)
    
    try:
        # Save uploaded file
        file_path = session_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Perform transcription
        result = get_transcripts_and_raw_times(file_path.as_posix(), session_dir)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/transcribe/srt")
async def transcribe_to_srt(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to transcribe")
):
    """
    Transcribe an audio file and return SRT subtitle format
    
    - **file**: Audio file (supported formats: wav, mp3, flac, etc.)
    
    Returns SRT formatted subtitle file
    """
    # Get transcription result
    result = await transcribe_audio(background_tasks, file)
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)
    
    # Convert to SRT format
    segment_timestamps = [
        {"start": seg.start, "end": seg.end, "segment": seg.text}
        for seg in result.segments
    ]
    
    srt_content = generate_srt_content(segment_timestamps)
    
    # Create temporary SRT file
    session_id = str(uuid.uuid4())
    session_dir = Path(tempfile.gettempdir()) / f"srt_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    srt_file_path = session_dir / f"{Path(file.filename).stem}.srt"
    with open(srt_file_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_session_dir, session_dir)
    
    return FileResponse(
        path=srt_file_path,
        filename=f"{Path(file.filename).stem}.srt",
        media_type="text/plain"
    )

if __name__ == "__main__":
    print("Starting Speech Transcription API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)