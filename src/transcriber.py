"""Speech-to-text transcription module using Whisper."""

import whisper
import config


class Transcriber:
    """Transcribes audio segments to text using OpenAI Whisper."""
    
    def __init__(self, model_size: str = None):
        """Initialize the Whisper model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large).
                       If None, uses config default.
        """
        model_size = model_size or config.WHISPER_MODEL_SIZE
        self.model = whisper.load_model(model_size, device=config.WHISPER_DEVICE)
        self.model_size = model_size
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> dict:
        """Transcribe a single audio file.
        
        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., 'en', 'es'). If None, auto-detect.
            
        Returns:
            Dictionary containing transcription results with 'text', 'segments', etc.
        """
        try:
            result = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe"
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {e}")
    
    def transcribe_segments(self, segments: list, language: str = None) -> list:
        """Transcribe multiple audio segments.
        
        Args:
            segments: List of segment dictionaries with 'path', 'start', 'end' keys.
            language: Language code for transcription.
            
        Returns:
            List of dictionaries with transcription results and timing info.
        """
        results = []
        
        for segment in segments:
            print(f"Transcribing segment {segment['index'] + 1}/{len(segments)}...")
            
            transcription = self.transcribe_audio(segment['path'], language)
            
            results.append({
                'index': segment['index'],
                'start': segment['start'],
                'end': segment['end'],
                'text': transcription['text'].strip(),
                'language': transcription.get('language', language)
            })
        
        return results
