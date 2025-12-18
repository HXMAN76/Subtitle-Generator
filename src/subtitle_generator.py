"""Subtitle generation module for creating SRT and VTT files."""

from pathlib import Path
from datetime import timedelta
import config


class SubtitleGenerator:
    """Generates subtitle files in various formats."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the subtitle generator.
        
        Args:
            output_dir: Directory to save subtitle files.
        """
        self.output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
    
    @staticmethod
    def format_timestamp_srt(seconds: float) -> str:
        """Format seconds to SRT timestamp format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds.
            
        Returns:
            Formatted timestamp string.
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def format_timestamp_vtt(seconds: float) -> str:
        """Format seconds to WebVTT timestamp format (HH:MM:SS.mmm).
        
        Args:
            seconds: Time in seconds.
            
        Returns:
            Formatted timestamp string.
        """
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millis = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def generate_srt(self, subtitles: list, output_filename: str) -> str:
        """Generate an SRT subtitle file.
        
        Args:
            subtitles: List of subtitle dictionaries with 'start', 'end', 'text' keys.
            output_filename: Name of the output file.
            
        Returns:
            Path to the generated SRT file.
        """
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, subtitle in enumerate(subtitles, start=1):
                start_time = self.format_timestamp_srt(subtitle['start'])
                end_time = self.format_timestamp_srt(subtitle['end'])
                text = subtitle['text']
                
                f.write(f"{idx}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        print(f"SRT file generated: {output_path}")
        return str(output_path)
    
    def generate_vtt(self, subtitles: list, output_filename: str) -> str:
        """Generate a WebVTT subtitle file.
        
        Args:
            subtitles: List of subtitle dictionaries with 'start', 'end', 'text' keys.
            output_filename: Name of the output file.
            
        Returns:
            Path to the generated VTT file.
        """
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for idx, subtitle in enumerate(subtitles, start=1):
                start_time = self.format_timestamp_vtt(subtitle['start'])
                end_time = self.format_timestamp_vtt(subtitle['end'])
                text = subtitle['text']
                
                f.write(f"{idx}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n\n")
        
        print(f"VTT file generated: {output_path}")
        return str(output_path)
    
    def generate_subtitles(self, subtitles: list, output_filename: str, 
                          format: str = None) -> str:
        """Generate subtitle file in the specified format.
        
        Args:
            subtitles: List of subtitle dictionaries.
            output_filename: Base name for the output file (extension will be added).
            format: Subtitle format ('srt' or 'vtt'). If None, uses config default.
            
        Returns:
            Path to the generated subtitle file.
        """
        format = (format or config.SUBTITLE_FORMAT).lower()
        
        # Remove extension if provided
        base_name = Path(output_filename).stem
        
        if format == 'srt':
            return self.generate_srt(subtitles, f"{base_name}.srt")
        elif format == 'vtt':
            return self.generate_vtt(subtitles, f"{base_name}.vtt")
        else:
            raise ValueError(f"Unsupported subtitle format: {format}")
