"""
Translator Module - NMT Integration.

This module provides the Translator class that wraps the NMT subsystem
for use in the main application pipeline.

Supported target languages: as, bn, gu, hi, kn, ml, mr, or, pa, ta, te

Note: Requires a trained NMT model. Without a trained model,
translations will not be available.
"""

import os
from pathlib import Path
from typing import Optional, List

import torch

from src.nmt.languages import (
    SUPPORTED_LANGUAGES,
    get_all_language_tags,
    is_supported_language,
    get_language_name
)


class Translator:
    """Translator using Neural Machine Translation.
    
    This class provides a simple interface for translation that
    integrates with the subtitle generation pipeline.
    
    Args:
        model_dir: Directory containing model checkpoint and tokenizer.
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
        self.translator = None
        self.available = False
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load model
        self._load_model()
    
    def _load_model(self):
        """Load the NMT model if available."""
        checkpoint_path = self.model_dir / "best.pt"
        tokenizer_path = self.model_dir / "nmt_spm.model"
        
        if not checkpoint_path.exists():
            print(f"[Translator] No model checkpoint found at {checkpoint_path}")
            print("[Translator] Translation will not be available.")
            print("[Translator] Train a model with: python scripts/train_nmt.py")
            return
        
        if not tokenizer_path.exists():
            print(f"[Translator] No tokenizer found at {tokenizer_path}")
            return
        
        try:
            from src.nmt.inference import NMTTranslator
            from src.nmt.tokenizer import Tokenizer
            from src.nmt.model.transformer import Transformer
            
            # Load tokenizer with all language tags
            tokenizer = Tokenizer(
                model_path=str(tokenizer_path),
                language_tags=get_all_language_tags()
            )
            
            # Load model
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            config = checkpoint.get('config', {})
            config['vocab_size'] = tokenizer.vocab_size
            
            model = Transformer.from_config(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Create translator
            self.translator = NMTTranslator(
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                beam_size=self.beam_size
            )
            
            self.available = True
            print(f"[Translator] Model loaded successfully ({model.count_parameters_readable()} params)")
            
        except Exception as e:
            print(f"[Translator] Failed to load model: {e}")
            self.available = False
    
    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "hi"
    ) -> str:
        """Translate text.
        
        Args:
            text: Text to translate.
            source_lang: Source language code (always 'en').
            target_lang: Target language code (as, bn, gu, hi, kn, ml, mr, or, pa, ta, te).
        
        Returns:
            Translated text, or original text if translation unavailable.
        """
        if not self.available or self.translator is None:
            return text
        
        # Validate target language
        if not is_supported_language(target_lang):
            print(f"[Translator] Unsupported language: {target_lang}")
            print(f"[Translator] Supported: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}")
            return text
        
        try:
            return self.translator.translate(
                text,
                source_lang=f"<{source_lang}>",
                target_lang=f"<{target_lang}>"
            )
        except Exception as e:
            print(f"[Translator] Translation error: {e}")
            return text
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported target language codes."""
        return list(SUPPORTED_LANGUAGES.keys())
    
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
        if not self.available or self.translator is None:
            return texts
        
        try:
            return self.translator.translate_batch(
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
        if not self.available or self.translator is None:
            print("[Translator] Model not available, returning original text")
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
    
    def is_available(self) -> bool:
        """Check if translation is available."""
        return self.available
