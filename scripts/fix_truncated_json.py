#!/usr/bin/env python3
"""Fix truncated JSON file by finding last valid entry."""
import sys

def fix_truncated_json(filepath):
    print(f"Reading {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the last complete entry (ends with "}\n    }" or similar patterns)
    # Look for last occurrence of "}," followed by potential truncation
    last_valid = content.rfind('},')
    
    if last_valid == -1:
        print("Could not find any complete entries!")
        return
    
    # Truncate to last valid entry and close the array
    fixed_content = content[:last_valid+1] + '\n]'
    
    backup_path = filepath + '.backup'
    print(f"Backing up to {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Writing fixed JSON to {filepath}")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    # Verify
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ Fixed! File now has {len(data)} valid entries.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_truncated_json.py <filepath>")
        sys.exit(1)
    fix_truncated_json(sys.argv[1])
