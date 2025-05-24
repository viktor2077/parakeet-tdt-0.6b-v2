#!/usr/bin/env python3
"""
Simple test client for the Speech Transcription API
"""

import requests
import sys
import json
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        
        health_data = response.json()
        print(f"✅ Health check passed:")
        print(f"   Status: {health_data['status']}")
        print(f"   Model loaded: {health_data['model_loaded']}")
        print(f"   Device: {health_data['device']}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def transcribe_audio(audio_file_path: str):
    """Test audio transcription"""
    audio_path = Path(audio_file_path)
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    print(f"🎤 Transcribing audio file: {audio_path.name}")
    
    try:
        with open(audio_path, "rb") as audio_file:
            files = {"file": (audio_path.name, audio_file, "audio/wav")}
            
            response = requests.post(
                f"{API_BASE_URL}/transcribe",
                files=files,
                timeout=300  # 5 minutes timeout for long audio
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result["success"]:
                print(f"✅ Transcription successful!")
                print(f"   Duration: {result.get('duration', 'N/A')} seconds")
                print(f"   Segments: {len(result['segments'])}")
                print("\n📝 Transcription:")
                print("-" * 50)
                
                for i, segment in enumerate(result['segments'], 1):
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text']
                    print(f"{i:2d}. [{start_time:6.2f}s - {end_time:6.2f}s]: {text}")
                
                return True
            else:
                print(f"❌ Transcription failed: {result.get('message', 'Unknown error')}")
                return False
                
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def transcribe_to_srt(audio_file_path: str, output_srt_path: str = None):
    """Test SRT subtitle generation"""
    audio_path = Path(audio_file_path)
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    if not output_srt_path:
        output_srt_path = audio_path.with_suffix('.srt')
    
    print(f"📺 Generating SRT subtitles for: {audio_path.name}")
    
    try:
        with open(audio_path, "rb") as audio_file:
            files = {"file": (audio_path.name, audio_file, "audio/wav")}
            
            response = requests.post(
                f"{API_BASE_URL}/transcribe/srt",
                files=files,
                timeout=300
            )
            response.raise_for_status()
            
            # Save SRT content
            with open(output_srt_path, "wb") as srt_file:
                srt_file.write(response.content)
            
            print(f"✅ SRT file saved: {output_srt_path}")
            
            # Show preview of SRT content
            with open(output_srt_path, "r", encoding="utf-8") as srt_file:
                content = srt_file.read()
                preview = content[:500] + "..." if len(content) > 500 else content
                print("\n📺 SRT Preview:")
                print("-" * 30)
                print(preview)
            
            return True
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Speech Transcription API Test Client")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("❌ Cannot proceed - API is not healthy")
        sys.exit(1)
    
    # Check for audio file argument
    if len(sys.argv) < 2:
        print("\n📖 Usage:")
        print(f"   python {sys.argv[0]} <audio_file_path>")
        print("\n💡 Example:")
        print(f"   python {sys.argv[0]} test_audio.wav")
        print("\n🎵 Supported formats: WAV, MP3, FLAC, OGG, MP4")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print("\n" + "=" * 50)
    
    # Test transcription
    success1 = transcribe_audio(audio_file)
    
    print("\n" + "=" * 50)
    
    # Test SRT generation
    success2 = transcribe_to_srt(audio_file)
    
    print("\n" + "=" * 50)
    
    if success1 and success2:
        print("🎉 All tests passed successfully!")
    else:
        print("⚠️  Some tests failed. Check the output above.")

if __name__ == "__main__":
    main() 