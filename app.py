"""Main application entry point for Subtitle Generator and Translator."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.audio_processor import AudioProcessor
from src.transcriber import Transcriber
from src.translator import Translator
from src.subtitle_generator import SubtitleGenerator
import config


class SubtitleApp:
    """Main application class for subtitle generation and translation.
    
    Uses full audio mode for faster processing - no VAD/segmentation needed
    as faster-whisper handles this internally with streaming.
    """
    
    def __init__(self):
        """Initialize all components."""
        print("Initializing Subtitle Generator...")
        
        self.audio_processor = AudioProcessor()
        self.transcriber = Transcriber()
        self.translator = Translator()
        self.subtitle_generator = SubtitleGenerator()
        
        print("Initialization complete!")
    
    def process_video(self, video_path: str, translate: bool = False, 
                     output_name: str = None) -> dict:
        """Process a video file to generate subtitles.
        
        Uses full audio mode - transcribes entire audio in one pass
        for maximum speed and efficiency.
        
        Args:
            video_path: Path to the input video file.
            translate: Whether to translate subtitles.
            output_name: Base name for output files (optional).
            
        Returns:
            Dictionary with paths to generated files.
        """
        video_name = Path(video_path).stem
        output_name = output_name or video_name
        
        print("\n" + "="*60)
        print(f"Processing video: {video_name}")
        print("="*60)
        
        # Step 1: Extract audio
        print("\n[1/3] Extracting audio from video...")
        audio_path = self.audio_processor.convert_video_to_audio(video_path)
        print(f"✓ Audio extracted: {audio_path}")
        
        # Step 2: Transcribe FULL audio (faster-whisper handles VAD internally)
        print("\n[2/3] Transcribing audio (full audio mode)...")
        transcriptions = self.transcriber.transcribe_full_audio(
            audio_path, 
            language=config.SOURCE_LANGUAGE
        )
        print(f"✓ Transcribed {len(transcriptions)} segments")
        
        # Step 3: Generate subtitles
        print("\n[3/3] Generating subtitle files...")
        
        results = {}
        
        # Generate original subtitles
        original_subtitle_path = self.subtitle_generator.generate_subtitles(
            transcriptions,
            f"{output_name}_original",
            format=config.SUBTITLE_FORMAT
        )
        results['original_subtitles'] = original_subtitle_path
        print(f"✓ Original subtitles: {original_subtitle_path}")
        
        # Translate if requested
        if translate:
            print("\nTranslating subtitles...")
            translated_transcriptions = self.translator.translate_subtitles(transcriptions)
            
            translated_subtitle_path = self.subtitle_generator.generate_subtitles(
                translated_transcriptions,
                f"{output_name}_{config.TARGET_LANGUAGE}",
                format=config.SUBTITLE_FORMAT
            )
            results['translated_subtitles'] = translated_subtitle_path
            print(f"✓ Translated subtitles: {translated_subtitle_path}")
        
        print("\n" + "="*60)
        print("Processing complete!")
        print("="*60)
        
        return results


def main():
    """Main function."""
    print("=" * 60)
    print("Subtitle Generator and Translator")
    print("=" * 60)
    
    # Initialize application
    app = SubtitleApp()
    
    # Example usage - update this path to your video file
    video_path = "./examples/sample_video.mp4"
    
    if not Path(video_path).exists():
        print(f"\nError: Video file not found: {video_path}")
        print("\nUsage: Place your video file in the project directory and update the video_path variable.")
        return
    
    # Process video
    results = app.process_video(
        video_path=video_path,
        translate=True  # Set to False if you don't want translation
    )
    
    print("\nGenerated files:")
    for key, path in results.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()