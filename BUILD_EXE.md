# Building Subtitle Generator as .exe

This guide explains how to package the Subtitle Generator as a standalone Windows executable that runs completely offline.

## Prerequisites

1. Windows OS (for building Windows .exe)
2. Python 3.8+ installed
3. All dependencies installed

## Method 1: Using PyInstaller (Recommended)

### Step 1: Install PyInstaller

```bash
pip install pyinstaller
```

### Step 2: Pre-download Models

Before building, download all required models so they're included in the .exe:

```python
# Run this script once to download models
import whisper
import torch

# Download Whisper model
print("Downloading Whisper model...")
whisper.load_model("tiny")  # or "base", "small", "medium", "large"

# Download Silero VAD
print("Downloading Silero VAD...")
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

print("Models downloaded successfully!")
```

### Step 3: Create PyInstaller Spec File

Create a file named `subtitle_generator.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('models', 'models'),
        ('config.py', '.'),
    ],
    hiddenimports=[
        'whisper',
        'torch',
        'moviepy',
        'pydub',
        'numpy',
        'scipy',
        'tiktoken',
        'regex',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SubtitleGenerator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for GUI version
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico'  # Optional: add your icon
)
```

### Step 4: Build the Executable

```bash
pyinstaller subtitle_generator.spec
```

Or use the simple one-liner:

```bash
pyinstaller --onefile --name SubtitleGenerator --add-data "src:src" --add-data "config.py:." --add-data "models:models" --hidden-import=whisper --hidden-import=torch app.py
```

### Step 5: Include FFmpeg

1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract `ffmpeg.exe`, `ffprobe.exe` from the archive
3. Place them in the same directory as `SubtitleGenerator.exe`

### Step 6: Distribution Package

Create a folder with:
```
SubtitleGenerator/
├── SubtitleGenerator.exe
├── ffmpeg.exe
├── ffprobe.exe
├── models/
│   └── translation/
│       ├── model.pt (if trained)
│       └── vocab.json (if trained)
├── output/ (empty folder)
├── temp/ (empty folder)
└── README.txt (usage instructions)
```

## Method 2: Using Auto-py-to-exe (GUI Method)

### Step 1: Install Auto-py-to-exe

```bash
pip install auto-py-to-exe
```

### Step 2: Launch GUI

```bash
auto-py-to-exe
```

### Step 3: Configure Settings

1. **Script Location**: Select `app.py`
2. **Onefile**: Select "One File"
3. **Console**: Select "Console Based" (or "Window Based" for GUI)
4. **Additional Files**:
   - Add folder: `src`
   - Add file: `config.py`
   - Add folder: `models`
5. **Hidden Imports**: Add `whisper`, `torch`, `moviepy`, `pydub`
6. Click "Convert .py to .exe"

## Testing the Executable

1. Copy the .exe to a clean directory
2. Add FFmpeg executables
3. Place a test video file
4. Run the executable
5. Check the `output/` folder for generated subtitles

## Optimizations

### Reduce File Size

1. Use smaller Whisper model (tiny or base)
2. Use UPX compression:
   ```bash
   pyinstaller --onefile --upx-dir=/path/to/upx app.py
   ```

### Faster Startup

Use --noupx if startup time is critical:
```bash
pyinstaller --onefile --noupx app.py
```

### Include Only CPU Version of PyTorch

To reduce size, install CPU-only PyTorch before building:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Common Issues and Solutions

### Issue 1: "Failed to load model"

**Solution**: Ensure models are pre-downloaded and included in the build:
- Check the `models/` folder is included in `datas`
- Verify torch.hub cache is accessible

### Issue 2: "FFmpeg not found"

**Solution**: 
- Include ffmpeg.exe in the same directory
- Or modify config.py to specify FFmpeg path

### Issue 3: Large file size (>500MB)

**Solution**:
- Use CPU-only PyTorch
- Use smaller Whisper model
- Enable UPX compression
- Use PyInstaller's --exclude-module for unused packages

### Issue 4: Slow startup

**Solution**:
- Use --noupx
- Pre-load models in __init__ methods
- Consider using a splash screen

## Advanced: Creating an Installer

### Using Inno Setup (Windows)

1. Download Inno Setup: https://jrsoftware.org/isinfo.php
2. Create an installer script:

```ini
[Setup]
AppName=Subtitle Generator
AppVersion=1.0
DefaultDirName={pf}\SubtitleGenerator
DefaultGroupName=Subtitle Generator
OutputDir=installer

[Files]
Source: "dist\SubtitleGenerator.exe"; DestDir: "{app}"
Source: "ffmpeg.exe"; DestDir: "{app}"
Source: "ffprobe.exe"; DestDir: "{app}"
Source: "models\*"; DestDir: "{app}\models"; Flags: recursesubdirs

[Icons]
Name: "{group}\Subtitle Generator"; Filename: "{app}\SubtitleGenerator.exe"
```

3. Compile the script to create an installer

## Distribution Checklist

- [ ] Test on clean Windows machine
- [ ] Include README with usage instructions
- [ ] Include license information
- [ ] Verify all dependencies are included
- [ ] Test with different video formats
- [ ] Ensure FFmpeg is included
- [ ] Test translation feature (if model is trained)
- [ ] Create backup of original files
- [ ] Version your releases

## Performance Tips

1. **Smaller Models**: Use "tiny" or "base" Whisper for faster processing
2. **Batch Processing**: Process multiple videos in sequence
3. **Disk Space**: Ensure temp/ directory has enough space
4. **RAM**: Close other applications for large video files

## Troubleshooting Build Errors

If build fails, try:

```bash
# Clean previous builds
pyinstaller --clean subtitle_generator.spec

# Verbose output for debugging
pyinstaller --log-level DEBUG subtitle_generator.spec

# Manual dependency check
python -c "import whisper, torch, moviepy, pydub; print('All imports OK')"
```

## Example Distribution Structure

```
SubtitleGenerator_v1.0/
├── SubtitleGenerator.exe        # Main executable
├── ffmpeg.exe                   # Required
├── ffprobe.exe                  # Required
├── README.txt                   # Usage instructions
├── LICENSE.txt                  # License information
├── models/                      # Model files
│   └── translation/
│       ├── model.pt
│       └── vocab.json
└── examples/                    # Optional example files
    └── sample_video.mp4
```

---

**Note**: First run may take longer as Windows SmartScreen checks the executable. This is normal for unsigned executables.
