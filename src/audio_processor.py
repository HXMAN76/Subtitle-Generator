"""Audio extraction and processing module."""

from pathlib import Path
from moviepy import AudioFileClip
from pydub import AudioSegment
import config


class AudioProcessor:
    """Handles audio extraction and segmentation."""
    
    def __init__(self):
        """Initialize the audio processor."""
        self.temp_dir = config.TEMP_DIR
    
    def extract_audio_from_video(self, video_path: str, audio_path: str) -> None:
        """Extract audio from a video file and save it as an audio file.
        
        Args:
            video_path: Path to the input video file.
            audio_path: Path to save the extracted audio file.
        """
        try:
            video_clip = AudioFileClip(video_path)
            video_clip.write_audiofile(audio_path, bitrate=config.AUDIO_BITRATE)
            video_clip.close()
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from video: {e}")
    
    def convert_video_to_audio(self, video_path: str) -> str:
        """Convert video to audio and save in temp directory.
        
        Args:
            video_path: Path to the input video file.
            
        Returns:
            Path to the extracted audio file.
        """
        video_name = Path(video_path).stem
        audio_path = self.temp_dir / f"{video_name}.{config.AUDIO_FORMAT}"
        self.extract_audio_from_video(video_path, str(audio_path))
        return str(audio_path)
    
    def segment_audio(self, audio_path: str, speech_timestamps: list) -> list:
        """Segment audio based on speech timestamps.
        
        Args:
            audio_path: Path to the audio file.
            speech_timestamps: List of speech timestamp dictionaries with 'start' and 'end' keys.
            
        Returns:
            List of paths to segmented audio files.
        """
        audio = AudioSegment.from_file(audio_path)
        segments = []
        
        voice_dir = self.temp_dir / "voice"
        voice_dir.mkdir(exist_ok=True)
        
        for i, ts in enumerate(speech_timestamps):
            start_ms = int(ts['start'] * 1000)
            end_ms = int(ts['end'] * 1000)
            segment = audio[start_ms:end_ms]
            
            segment_path = voice_dir / f"segment_{i:04d}.{config.AUDIO_FORMAT}"
            segment.export(str(segment_path), format=config.AUDIO_FORMAT)
            segments.append({
                'path': str(segment_path),
                'start': ts['start'],
                'end': ts['end'],
                'index': i
            })
        
        return segments
