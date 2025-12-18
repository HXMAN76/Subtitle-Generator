"""Voice Activity Detection (VAD) module using Silero VAD."""

import torch
import config


class VoiceActivityDetector:
    """Detects speech segments in audio using Silero VAD."""
    
    def __init__(self):
        """Initialize the VAD model."""
        torch.set_num_threads(config.NUM_THREADS)
        
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
    
    def detect_speech(self, audio_path: str) -> list:
        """Detect speech timestamps in an audio file.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            List of dictionaries containing 'start' and 'end' timestamps in seconds.
        """
        try:
            wav = self.read_audio(audio_path)
            speech_timestamps = self.get_speech_timestamps(
                wav,
                self.model,
                threshold=config.VAD_THRESHOLD,
                min_speech_duration_ms=config.VAD_MIN_SPEECH_DURATION_MS,
                max_speech_duration_s=config.VAD_MAX_SPEECH_DURATION_S,
                min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
                return_seconds=True
            )
            return speech_timestamps
        except Exception as e:
            raise RuntimeError(f"Failed to detect speech: {e}")
