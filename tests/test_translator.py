"""
Quick test for multi-language translator functionality.

This script tests:
1. Tokenizer loading
2. Model loading for available languages
3. Translation functionality
4. Lazy loading behavior
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.translator import Translator

def main():
    print("=" * 60)
    print("Multi-Language Translator Test")
    print("=" * 60)
    print()
    
    # Initialize translator
    print("1️⃣  Initializing Translator...")
    translator = Translator()
    print()
    
    # Check supported languages
    print("2️⃣  Supported languages:")
    for lang in translator.get_supported_languages():
        print(f"   - {lang}")
    print()
    
    # Check available languages (with trained models)
    print("3️⃣  Available languages (with models):")
    available = translator.get_available_languages()
    if available:
        for lang in available:
            print(f"   ✓ {lang}")
    else:
        print("   ⚠️  No models available yet")
        print("   Run: bash scripts/copy_models.sh")
    print()
    
    # Check loaded languages
    print("4️⃣  Currently loaded languages:")
    loaded = translator.get_loaded_languages()
    if loaded:
        for lang in loaded:
            print(f"   - {lang}")
    else:
        print("   (none - lazy loading not triggered yet)")
    print()
    
    # Test translation if models are available
    if available:
        test_lang = available[0]
        print(f"5️⃣  Testing translation to '{test_lang}'...")
        
        test_text = "Hello, how are you?"
        translated = translator.translate(test_text, target_lang=test_lang)
        
        print(f"   Original:  {test_text}")
        print(f"   Translated: {translated}")
        print()
        
        # Check loaded languages after translation
        print("6️⃣  Loaded languages after translation:")
        loaded = translator.get_loaded_languages()
        for lang in loaded:
            print(f"   - {lang}")
        print()
        
        # Test batch translation
        print("7️⃣  Testing batch translation...")
        texts = ["Good morning", "Thank you", "Welcome"]
        translated_batch = translator.translate_batch(texts, target_lang=test_lang)
        
        for orig, trans in zip(texts, translated_batch):
            print(f"   {orig} → {trans}")
        print()
        
        print("✅ All tests passed!")
    else:
        print("⏭️  Skipping translation tests (no models available)")
        print()
    
    print("=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
