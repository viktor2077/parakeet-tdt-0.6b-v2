# Speech Transcription API

A FastAPI-based REST API service for speech-to-text transcription using NVIDIA's parakeet-tdt-0.6b-v2 model. This API provides high-quality English speech recognition with automatic punctuation, capitalization, and accurate word-level timestamps.

## Features

- 🎤 **High-Quality Transcription**: Uses NVIDIA's 600M parameter parakeet-tdt-0.6b-v2 model
- ⏱️ **Accurate Timestamps**: Provides word-level timing information
- 📝 **Multiple Output Formats**: JSON response or SRT subtitle format
- 🔧 **Automatic Audio Processing**: Handles resampling and channel conversion
- 🚀 **Long Audio Support**: Optimized settings for audio longer than 8 minutes
- 📊 **OpenAPI Compatible**: Full Swagger/OpenAPI documentation
- 🛡️ **Error Handling**: Comprehensive error handling and validation

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- FFmpeg (for audio processing)

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd parakeet-tdt-0.6b-v2
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the API server**
```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### 2. Audio Transcription
```http
POST /transcribe
```

**Parameters:**
- `file`: Audio file (multipart/form-data)

**Supported formats:** WAV, MP3, FLAC, OGG, MP4

**Response:**
```json
{
  "success": true,
  "segments": [
    {
      "start": 0.5,
      "end": 2.1,
      "text": "Hello, how are you today?"
    },
    {
      "start": 2.5,
      "end": 4.8,
      "text": "I'm doing great, thank you for asking."
    }
  ],
  "duration": 15.3,
  "message": "Transcription completed successfully"
}
```

#### 3. SRT Subtitle Generation
```http
POST /transcribe/srt
```

**Parameters:**
- `file`: Audio file (multipart/form-data)

**Response:** SRT file download

```srt
1
00:00:00,500 --> 00:00:02,100
Hello, how are you today?

2
00:00:02,500 --> 00:00:04,800
I'm doing great, thank you for asking.
```

## Usage Examples

### Python Client Example

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Transcribe audio file
with open("audio.wav", "rb") as f:
    files = {"file": ("audio.wav", f, "audio/wav")}
    response = requests.post("http://localhost:8000/transcribe", files=files)
    result = response.json()
    
    if result["success"]:
        for segment in result["segments"]:
            print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Transcribe audio
curl -X POST "http://localhost:8000/transcribe" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav"

# Get SRT subtitle file
curl -X POST "http://localhost:8000/transcribe/srt" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@audio.wav" \
     --output subtitles.srt
```

### JavaScript/Node.js Example

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function transcribeAudio(filePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    try {
        const response = await axios.post('http://localhost:8000/transcribe', form, {
            headers: form.getHeaders()
        });
        
        console.log('Transcription result:', response.data);
        return response.data;
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

transcribeAudio('audio.wav');
```

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Specify which GPU to use (default: auto-detect)
- `MODEL_CACHE_DIR`: Directory to cache the model files

### Model Configuration

The API automatically:
- Detects available hardware (CUDA/CPU)
- Loads the parakeet-tdt-0.6b-v2 model on startup
- Applies optimized settings for long audio (>8 minutes)
- Handles memory cleanup after each request

## Performance Considerations

### Hardware Requirements

- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended)
- **CPU**: Multi-core processor (fallback option)
- **RAM**: 8GB+ system memory
- **Storage**: 2GB+ for model cache

### Optimization Tips

1. **Use GPU**: Significantly faster than CPU processing
2. **Audio Format**: WAV files typically process fastest
3. **File Size**: For very long audio files (>3 hours), consider chunking
4. **Concurrent Requests**: API handles one request at a time to avoid memory issues

## Error Handling

The API provides detailed error messages for common issues:

- **400 Bad Request**: Unsupported file format
- **413 Payload Too Large**: File size exceeds limits
- **500 Internal Server Error**: Processing or model errors

## Development

### Running in Development Mode

```bash
# With auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Test with sample audio
curl -X POST "http://localhost:8000/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_audio.wav"
```

## License

This project uses the NVIDIA parakeet-tdt-0.6b-v2 model, which is available for both commercial and non-commercial use. Please refer to the [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) for detailed licensing information.

## Troubleshooting

### Common Issues

1. **Model loading fails**: Check CUDA installation and GPU memory
2. **Audio processing errors**: Ensure FFmpeg is installed
3. **Memory errors**: Reduce concurrent requests or use CPU mode

### Getting Help

- Check the API documentation at `/docs`
- Review error messages in the server logs
- Ensure all dependencies are properly installed

## References

- [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)
- [Parakeet Model Paper](https://arxiv.org/abs/2304.06795)
- [Fast Conformer Paper](https://arxiv.org/abs/2305.05084)