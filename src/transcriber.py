"""Speech-to-text transcription module using faster-whisper.

Uses CTranslate2-optimized Whisper for 3-4x faster transcription.
"""

from faster_whisper import WhisperModel
import config


class Transcriber:
    """Transcribes audio segments to text using faster-whisper.
    
    This is a drop-in replacement for the original Whisper-based transcriber,
    but with significantly improved performance (3-4x faster on GPU, 8x on CPU).
    """
    
    def __init__(self, model_size: str = None):
        """Initialize the faster-whisper model.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3).
                       If None, uses config default.
        """
        model_size = model_size or config.WHISPER_MODEL_SIZE
        
        # Determine compute type based on device
        if config.WHISPER_DEVICE == "cuda":
            compute_type = "float16"  # Faster on GPU
        else:
            compute_type = "int8"  # Faster on CPU
        
        self.model = WhisperModel(
            model_size,
            device=config.WHISPER_DEVICE,
            compute_type=compute_type
        )
        self.model_size = model_size
        print(f"[Transcriber] Loaded faster-whisper ({model_size}) on {config.WHISPER_DEVICE}")
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> dict:
        """Transcribe a single audio file.
        
        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., 'en', 'es'). If None, auto-detect.
            
        Returns:
            Dictionary containing transcription results with 'text', 'segments', etc.
        """
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                vad_filter=True,  # Use Silero VAD for better accuracy
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400
                )
            )
            
            # Collect all segments
            all_segments = []
            full_text = []
            
            for segment in segments:
                all_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'no_speech_prob': segment.no_speech_prob
                })
                full_text.append(segment.text.strip())
            
            return {
                'text': ' '.join(full_text),
                'segments': all_segments,
                'language': info.language,
                'language_probability': info.language_probability
            }
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
        total = len(segments)
        
        for segment in segments:
            idx = segment['index'] + 1
            print(f"Transcribing segment {idx}/{total}...", end='\r')
            
            transcription = self.transcribe_audio(segment['path'], language)
            
            results.append({
                'index': segment['index'],
                'start': segment['start'],
                'end': segment['end'],
                'text': transcription['text'].strip(),
                'language': transcription.get('language', language)
            })
        
        print()  # New line after progress
        return results
    
    def transcribe_full_audio(self, audio_path: str, language: str = None) -> list:
        """Transcribe full audio file and return timed segments.
        
        This is more efficient than transcribing individual segments
        as it processes the entire audio in one pass.
        
        Args:
            audio_path: Path to the full audio file.
            language: Language code for transcription.
            
        Returns:
            List of dictionaries with 'start', 'end', 'text' for each segment.
        """
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400
                )
            )
            
            results = []
            print(f"[Transcriber] Transcribing full audio (language: {info.language})...")
            
            for i, segment in enumerate(segments):
                results.append({
                    'index': i,
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'language': info.language,
                    'no_speech_prob': segment.no_speech_prob
                })
                print(f"  Processed {i+1} segments...", end='\r')
            
            print(f"\n[Transcriber] Transcribed {len(results)} segments")
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to transcribe audio: {e}")
