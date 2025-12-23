"""Utility functions for tests."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_test_video_path():
    """Get path to test video file if available."""
    examples_dir = Path(__file__).parent.parent / "examples"
    test_videos = list(examples_dir.glob("*.mp4"))
    
    if test_videos:
        return str(test_videos[0])
    return None


def get_test_audio_path():
    """Get path to test audio file if available."""
    examples_dir = Path(__file__).parent.parent / "examples"
    test_audio = list(examples_dir.glob("*.mp3")) + list(examples_dir.glob("*.wav"))
    
    if test_audio:
        return str(test_audio[0])
    return None


def create_temp_test_file(content: str, extension: str = ".txt") -> Path:
    """Create a temporary test file."""
    import tempfile
    
    fd, path = tempfile.mkstemp(suffix=extension)
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    
    return Path(path)


def cleanup_test_files(*paths):
    """Clean up test files."""
    for path in paths:
        if isinstance(path, (str, Path)):
            path = Path(path)
            if path.exists():
                path.unlink()
