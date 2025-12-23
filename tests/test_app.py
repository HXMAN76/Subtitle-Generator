"""Unit tests for Subtitle Generator components."""

import sys
import unittest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class TestConfig(unittest.TestCase):
    """Test configuration settings."""
    
    def test_directories_exist(self):
        """Test that required directories are created."""
        self.assertTrue(config.TEMP_DIR.exists())
        self.assertTrue(config.OUTPUT_DIR.exists())
        self.assertTrue(config.MODELS_DIR.exists())
        self.assertTrue(config.DATA_DIR.exists())
    
    def test_whisper_model_size_valid(self):
        """Test that Whisper model size is valid."""
        valid_sizes = ['tiny', 'base', 'small', 'medium', 'large']
        self.assertIn(config.WHISPER_MODEL_SIZE, valid_sizes)
    
    def test_whisper_device_valid(self):
        """Test that Whisper device is valid."""
        valid_devices = ['cpu', 'cuda']
        self.assertIn(config.WHISPER_DEVICE, valid_devices)
    
    def test_subtitle_format_valid(self):
        """Test that subtitle format is valid."""
        valid_formats = ['srt', 'vtt']
        self.assertIn(config.SUBTITLE_FORMAT, valid_formats)
    
    def test_vad_threshold_range(self):
        """Test that VAD threshold is in valid range."""
        self.assertGreaterEqual(config.VAD_THRESHOLD, 0.0)
        self.assertLessEqual(config.VAD_THRESHOLD, 1.0)


class TestAudioProcessor(unittest.TestCase):
    """Test AudioProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.audio_processor import AudioProcessor
        self.audio_processor = AudioProcessor()
    
    def test_audio_processor_init(self):
        """Test AudioProcessor initialization."""
        self.assertIsNotNone(self.audio_processor)
    
    def test_temp_dir_exists(self):
        """Test that temp directory is configured."""
        self.assertTrue(self.audio_processor.temp_dir.exists())


class TestVoiceActivityDetector(unittest.TestCase):
    """Test VoiceActivityDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.vad import VoiceActivityDetector
        self.vad = VoiceActivityDetector()
    
    def test_vad_init(self):
        """Test VAD initialization."""
        self.assertIsNotNone(self.vad)
        self.assertIsNotNone(self.vad.model)


class TestTranscriber(unittest.TestCase):
    """Test Transcriber class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.transcriber import Transcriber
        self.transcriber = Transcriber()
    
    def test_transcriber_init(self):
        """Test Transcriber initialization."""
        self.assertIsNotNone(self.transcriber)
        self.assertIsNotNone(self.transcriber.model)


class TestTranslator(unittest.TestCase):
    """Test Translator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.translator import Translator
        self.translator = Translator()
    
    def test_translator_init(self):
        """Test Translator initialization."""
        self.assertIsNotNone(self.translator)


class TestSubtitleGenerator(unittest.TestCase):
    """Test SubtitleGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from src.subtitle_generator import SubtitleGenerator
        self.subtitle_gen = SubtitleGenerator()
    
    def test_subtitle_generator_init(self):
        """Test SubtitleGenerator initialization."""
        self.assertIsNotNone(self.subtitle_gen)
    
    def test_format_timestamp_srt(self):
        """Test SRT timestamp formatting."""
        # Test converting seconds to SRT format
        result = self.subtitle_gen.format_timestamp(3661.5, format='srt')
        self.assertEqual(result, "01:01:01,500")
    
    def test_format_timestamp_vtt(self):
        """Test VTT timestamp formatting."""
        # Test converting seconds to VTT format
        result = self.subtitle_gen.format_timestamp(3661.5, format='vtt')
        self.assertEqual(result, "01:01:01.500")


class TestSubtitleApp(unittest.TestCase):
    """Test main SubtitleApp class."""
    
    def test_app_import(self):
        """Test that SubtitleApp can be imported."""
        from app import SubtitleApp
        self.assertIsNotNone(SubtitleApp)


if __name__ == '__main__':
    unittest.main()
