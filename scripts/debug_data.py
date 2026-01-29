
import json
import sys
from pathlib import Path

def inspect_jsonl(file_path):
    print(f"Inspecting: {file_path}")
    path = Path(file_path)
    if not path.exists():
        print("❌ File not found!")
        return

    with open(path, 'r', encoding='utf-8') as f:
        print(f"File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Read first 5 lines
        for i, line in enumerate(f):
            if i >= 5: break
            
            print(f"\n--- Line {i+1} ---")
            print(f"Raw content: {line.strip()[:200]}...")
            
            try:
                data = json.loads(line.strip())
                print(f"Parsed keys: {list(data.keys())}")
                
                # Check for expected keys
                src = data.get('src')
                tgt = data.get('tgt')
                print(f"src: {str(src)[:50]}...")
                print(f"tgt: {str(tgt)[:50]}...")
                
                if not src or not tgt:
                    print("⚠️ WARNING: 'src' or 'tgt' field is empty or missing!")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON Decode Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Default to checking one of the failing files if no args provided
        files = ["data/raw/train-en-ml.jsonl", "data/raw/train-en-hi.jsonl"]
    
    for f in files:
        inspect_jsonl(f)
