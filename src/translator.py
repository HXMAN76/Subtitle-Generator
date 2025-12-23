"""
Translator Module - NMT Integration.

This module provides the Translator class that wraps the NMT subsystem
for use in the main application pipeline.

Note: Requires a trained NMT model. Without a trained model,
translations will not be available.
"""

import os
from pathlib import Path
from typing import Optional

import torch


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
            
            # Load tokenizer
            tokenizer = Tokenizer(
                model_path=str(tokenizer_path),
                language_tags=["<en>", "<hi>"]
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
            source_lang: Source language code.
            target_lang: Target language code.
        
        Returns:
            Translated text, or original text if translation unavailable.
        """
        if not self.available or self.translator is None:
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
    
    def is_available(self) -> bool:
        """Check if translation is available."""
        return self.available
