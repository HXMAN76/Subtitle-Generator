"""
Translator Module - Multi-Language NMT Integration.

This module provides the Translator class that wraps the NMT subsystem
for use in the main application pipeline.

Supports lazy loading of per-language models for memory efficiency.

Supported target languages: as, bn, gu, hi, kn, ml, mr, or, pa, ta, te
"""

import os
from pathlib import Path
from typing import Optional, List, Dict

import torch

from src.nmt.languages import (
    SUPPORTED_LANGUAGES,
    get_all_language_tags,
    is_supported_language,
    get_language_name
)


class Translator:
    """Multi-language Translator using Neural Machine Translation.
    
    This class provides a simple interface for translation that
    integrates with the subtitle generation pipeline.
    
    Features:
    - Lazy loading: Models are loaded on-demand when first used
    - Multi-language: Supports all trained Indic language models
    - Shared tokenizer: One tokenizer for all language models
    
    Directory structure expected:
        models/translation/
        ├── nmt_spm.model       # Shared tokenizer
        ├── as/best.pt          # Assamese model
        ├── bn/best.pt          # Bengali model
        └── ...                 # Other language models
    
    Args:
        model_dir: Directory containing tokenizer and language model folders.
        device: Device for inference ('cuda', 'cpu', or None for auto).
        beam_size: Beam size for decoding (0 for greedy).
    """
    
    def __init__(
        self,
        model_dir: str = "models/translation",
        device: Optional[str] = None,
        beam_size: int = 4
    ):
        self.model_dir = Path(model_dir)
        self.beam_size = beam_size
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Shared tokenizer (loaded once)
        self.tokenizer = None
        self._tokenizer_loaded = False
        
        # Per-language model cache {lang_code: NMTTranslator}
        self._models: Dict[str, 'NMTTranslator'] = {}
        
        # Track available languages
        self._available_languages: Optional[List[str]] = None
        
        # Load tokenizer on init
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load the shared tokenizer."""
        tokenizer_path = self.model_dir / "nmt_spm.model"
        
        if not tokenizer_path.exists():
            print(f"[Translator] No tokenizer found at {tokenizer_path}")
            print("[Translator] Translation will not be available.")
            return
        
        try:
            from src.nmt.tokenizer import Tokenizer
            
            self.tokenizer = Tokenizer(
                model_path=str(tokenizer_path),
                language_tags=get_all_language_tags()
            )
            self._tokenizer_loaded = True
            print(f"[Translator] Tokenizer loaded (vocab_size={self.tokenizer.vocab_size})")
            
        except Exception as e:
            print(f"[Translator] Failed to load tokenizer: {e}")
            self._tokenizer_loaded = False
    
    def _load_model(self, target_lang: str) -> Optional['NMTTranslator']:
        """Load model for specific language (lazy loading).
        
        Args:
            target_lang: Target language code (e.g., 'hi', 'ta').
            
        Returns:
            NMTTranslator instance, or None if unavailable.
        """
        # Return cached model if available
        if target_lang in self._models:
            return self._models[target_lang]
        
        # Check if tokenizer is available
        if not self._tokenizer_loaded or self.tokenizer is None:
            print(f"[Translator] Cannot load model - tokenizer not available")
            return None
        
        # Check if model exists
        checkpoint_path = self.model_dir / target_lang / "best.pt"
        if not checkpoint_path.exists():
            print(f"[Translator] No model found for '{target_lang}' at {checkpoint_path}")
            return None
        
        try:
            from src.nmt.inference import NMTTranslator
            from src.nmt.model.transformer import Transformer
            
            print(f"[Translator] Loading {target_lang} model...")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get config and ensure vocab_size matches tokenizer
            config = checkpoint.get('config', {})
            config['vocab_size'] = self.tokenizer.vocab_size
            
            # Create and load model
            model = Transformer.from_config(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Create translator
            translator = NMTTranslator(
                model=model,
                tokenizer=self.tokenizer,
                device=self.device,
                beam_size=self.beam_size
            )
            
            # Cache the model
            self._models[target_lang] = translator
            print(f"[Translator] {get_language_name(target_lang)} model loaded ({model.count_parameters_readable()} params)")
            
            return translator
            
        except Exception as e:
            print(f"[Translator] Failed to load {target_lang} model: {e}")
            return None
    
    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "hi"
    ) -> str:
        """Translate text to target language.
        
        Args:
            text: Text to translate.
            source_lang: Source language code (typically 'en').
            target_lang: Target language code (as, bn, gu, hi, kn, ml, mr, or, pa, ta, te).
        
        Returns:
            Translated text, or original text if translation unavailable.
        """
        # Validate target language
        if not is_supported_language(target_lang):
            print(f"[Translator] Unsupported language: {target_lang}")
            print(f"[Translator] Supported: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}")
            return text
        
        # Get or load model for target language
        translator = self._load_model(target_lang)
        if translator is None:
            return text
        
        try:
            return translator.translate(
                text,
                source_lang=f"<{source_lang}>",
                target_lang=f"<{target_lang}>"
            )
        except Exception as e:
            print(f"[Translator] Translation error: {e}")
            return text
    
    def get_supported_languages(self) -> List[str]:
        """Get list of all supported target language codes."""
        return list(SUPPORTED_LANGUAGES.keys())
    
    def get_available_languages(self) -> List[str]:
        """Get list of languages with trained models available.
        
        Returns:
            List of language codes that have trained models.
        """
        if self._available_languages is not None:
            return self._available_languages
        
        available = []
        for lang in SUPPORTED_LANGUAGES.keys():
            checkpoint = self.model_dir / lang / "best.pt"
            if checkpoint.exists():
                available.append(lang)
        
        self._available_languages = available
        return available
    
    def get_loaded_languages(self) -> List[str]:
        """Get list of currently loaded model languages.
        
        Returns:
            List of language codes for models currently in memory.
        """
        return list(self._models.keys())
    
    def translate_batch(
        self,
        texts: list,
        source_lang: str = "en",
        target_lang: str = "hi"
    ) -> list:
        """Translate a batch of texts.
        
        Args:
            texts: List of texts to translate.
            source_lang: Source language code.
            target_lang: Target language code.
        
        Returns:
            List of translated texts.
        """
        # Validate target language
        if not is_supported_language(target_lang):
            print(f"[Translator] Unsupported language: {target_lang}")
            return texts
        
        # Get or load model
        translator = self._load_model(target_lang)
        if translator is None:
            return texts
        
        try:
            return translator.translate_batch(
                texts,
                source_lang=f"<{source_lang}>",
                target_lang=f"<{target_lang}>"
            )
        except Exception as e:
            print(f"[Translator] Batch translation error: {e}")
            return texts
    
    def translate_subtitles(
        self,
        transcriptions: list,
        source_lang: str = "en",
        target_lang: str = "hi"
    ) -> list:
        """Translate subtitle transcriptions.
        
        Args:
            transcriptions: List of dicts with 'text', 'start', 'end' keys.
            source_lang: Source language code.
            target_lang: Target language code.
        
        Returns:
            List of transcriptions with translated text.
        """
        # Check if we can translate
        if not self._tokenizer_loaded:
            print("[Translator] Tokenizer not available, returning original text")
            return transcriptions
        
        translator = self._load_model(target_lang)
        if translator is None:
            print(f"[Translator] Model for '{target_lang}' not available, returning original text")
            return transcriptions
        
        # Extract texts for batch translation
        texts = [t.get('text', '') for t in transcriptions]
        
        # Batch translate for efficiency
        translated_texts = self.translate_batch(texts, source_lang, target_lang)
        
        # Reconstruct transcriptions with translated text
        translated_transcriptions = []
        for original, translated_text in zip(transcriptions, translated_texts):
            translated_transcriptions.append({
                'text': translated_text,
                'start': original.get('start'),
                'end': original.get('end'),
                'original_text': original.get('text', '')  # Keep original for reference
            })
        
        return translated_transcriptions
    
    def is_available(self, target_lang: Optional[str] = None) -> bool:
        """Check if translation is available.
        
        Args:
            target_lang: Optional language to check. If None, checks if
                         any translation is available.
        
        Returns:
            True if translation is available.
        """
        if not self._tokenizer_loaded:
            return False
        
        if target_lang is None:
            # Check if any models are available
            return len(self.get_available_languages()) > 0
        
        # Check specific language
        checkpoint = self.model_dir / target_lang / "best.pt"
        return checkpoint.exists()
    
    def unload_model(self, target_lang: str) -> bool:
        """Unload a specific model to free memory.
        
        Args:
            target_lang: Language code of model to unload.
            
        Returns:
            True if model was unloaded, False if not loaded.
        """
        if target_lang in self._models:
            del self._models[target_lang]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"[Translator] Unloaded {target_lang} model")
            return True
        return False
    
    def unload_all_models(self):
        """Unload all models to free memory."""
        count = len(self._models)
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[Translator] Unloaded {count} models")
