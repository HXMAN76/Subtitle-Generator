"""
Supported Languages for NMT Translation.

Defines all supported Indic languages from the Samanantar dataset.
Source: https://huggingface.co/datasets/ai4bharat/samanantar

The model translates from English to any of these 11 Indic languages.
"""

from typing import Dict, List


# Supported target languages (English is always the source)
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
}

# Dravidian languages (benefit from unigram tokenization)
DRAVIDIAN_LANGUAGES = {'kn', 'ml', 'ta', 'te'}

# Language code to approximate sentence count in Samanantar
LANGUAGE_SIZES: Dict[str, str] = {
    "as": "140K",
    "bn": "8.5M",
    "gu": "3.1M",
    "hi": "8.6M",
    "kn": "4.0M",
    "ml": "5.8M",
    "mr": "3.6M",
    "or": "1.0M",
    "pa": "2.4M",
    "ta": "5.3M",
    "te": "4.8M",
}

# Source language (always English for this system)
SOURCE_LANGUAGE = "en"

# Default target language
DEFAULT_TARGET_LANGUAGE = "hi"


def get_language_tag(lang_code: str) -> str:
    """Get the language tag for a language code.
    
    Args:
        lang_code: Two-letter language code (e.g., 'en', 'hi').
    
    Returns:
        Language tag in format '<code>' (e.g., '<en>', '<hi>').
    """
    return f"<{lang_code}>"


def get_all_language_tags() -> List[str]:
    """Get all language tags for tokenizer training.
    
    Returns:
        List of all language tags, starting with English.
    """
    tags = [get_language_tag(SOURCE_LANGUAGE)]
    tags.extend(get_language_tag(code) for code in SUPPORTED_LANGUAGES)
    return tags


def get_language_name(lang_code: str) -> str:
    """Get the full name of a language from its code.
    
    Args:
        lang_code: Two-letter language code.
    
    Returns:
        Full language name, or 'English' for 'en', or 'Unknown' if not found.
    """
    if lang_code == "en":
        return "English"
    return SUPPORTED_LANGUAGES.get(lang_code, "Unknown")


def is_supported_language(lang_code: str) -> bool:
    """Check if a language code is supported for translation.
    
    Args:
        lang_code: Two-letter language code.
    
    Returns:
        True if the language is supported as a target language.
    """
    return lang_code in SUPPORTED_LANGUAGES


def validate_language(lang_code: str) -> None:
    """Validate that a language code is supported.
    
    Args:
        lang_code: Two-letter language code.
    
    Raises:
        ValueError: If the language is not supported.
    """
    if not is_supported_language(lang_code):
        supported = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
        raise ValueError(
            f"Unsupported language: '{lang_code}'. "
            f"Supported languages: {supported}"
        )


if __name__ == "__main__":
    # Print language information when run directly
    print("=" * 50)
    print("Supported Languages for NMT Translation")
    print("=" * 50)
    print(f"\nSource Language: English ({SOURCE_LANGUAGE})")
    print(f"\nTarget Languages ({len(SUPPORTED_LANGUAGES)}):")
    print("-" * 40)
    
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        size = LANGUAGE_SIZES.get(code, "?")
        print(f"  {code}: {name:12} (~{size} sentences)")
    
    print(f"\nLanguage Tags for Tokenizer:")
    print(f"  {get_all_language_tags()}")
