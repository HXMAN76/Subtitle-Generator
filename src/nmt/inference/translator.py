"""
High-Level Translation API.

Provides a simple interface for translation that handles:
- Tokenization
- Decoding (greedy or beam search)
- Detokenization
- Batch translation
"""

import torch
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from .beam_search import beam_search, BeamSearchDecoder
from .greedy import greedy_decode


class NMTTranslator:
    """High-level translator interface.
    
    Wraps the Transformer model with tokenization and decoding
    for easy translation of text.
    
    Args:
        model: Trained Transformer model.
        tokenizer: Tokenizer instance.
        device: Device for inference.
        beam_size: Beam size for beam search (0 for greedy).
        max_length: Maximum output length.
        length_penalty: Length penalty for beam search.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[torch.device] = None,
        beam_size: int = 4,
        max_length: int = 256,
        length_penalty: float = 0.6
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Cache token IDs
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id
    
    def translate(
        self,
        text: str,
        source_lang: str = "<en>",
        target_lang: str = "<hi>",
        return_scores: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Translate a single text.
        
        Args:
            text: Source text to translate.
            source_lang: Source language tag.
            target_lang: Target language tag.
            return_scores: Whether to return translation scores.
        
        Returns:
            Translated text, or dict with text and score if return_scores=True.
        """
        translations = self.translate_batch(
            [text],
            source_lang=source_lang,
            target_lang=target_lang,
            return_scores=return_scores
        )
        
        return translations[0]
    
    @torch.no_grad()
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "<en>",
        target_lang: str = "<hi>",
        return_scores: bool = False
    ) -> List[Union[str, Dict[str, Any]]]:
        """Translate a batch of texts.
        
        Args:
            texts: List of source texts.
            source_lang: Source language tag.
            target_lang: Target language tag.
            return_scores: Whether to return translation scores.
        
        Returns:
            List of translated texts or dicts with text and score.
        """
        # Tokenize source texts
        src_ids, _ = self.tokenizer.encode_batch(
            texts,
            max_length=self.max_length,
            add_bos=True,
            add_eos=True,
            add_lang_tag=source_lang
        )
        
        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=self.device)
        
        # Decode
        if self.beam_size > 1:
            decoder = BeamSearchDecoder(
                model=self.model,
                beam_size=self.beam_size,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                pad_id=self.pad_id
            )
            
            results = decoder.decode(src_tensor, n_best=1)
            
            translations = []
            for i, hyps in enumerate(results):
                best_hyp = hyps[0]
                decoded = self.tokenizer.decode(best_hyp.tokens, skip_special_tokens=True)
                
                if return_scores:
                    translations.append({
                        'text': decoded,
                        'score': best_hyp.score
                    })
                else:
                    translations.append(decoded)
        else:
            # Greedy decoding
            output = greedy_decode(
                self.model,
                src_tensor,
                bos_id=self.bos_id,
                eos_id=self.eos_id,
                pad_id=self.pad_id,
                max_length=self.max_length,
                device=self.device
            )
            
            translations = []
            for i in range(output.size(0)):
                tokens = output[i].tolist()
                decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
                
                if return_scores:
                    translations.append({
                        'text': decoded,
                        'score': 0.0  # Greedy doesn't have meaningful scores
                    })
                else:
                    translations.append(decoded)
        
        return translations
    
    def translate_subtitles(
        self,
        subtitles: List[Dict],
        source_lang: str = "<en>",
        target_lang: str = "<hi>"
    ) -> List[Dict]:
        """Translate a list of subtitle entries.
        
        Designed for integration with the larger subtitle processing system.
        
        Args:
            subtitles: List of subtitle dicts with 'text' key.
            source_lang: Source language tag.
            target_lang: Target language tag.
        
        Returns:
            List of subtitle dicts with translated text.
        """
        if not subtitles:
            return []
        
        # Extract texts
        texts = [sub['text'] for sub in subtitles]
        
        # Translate in batch
        translations = self.translate_batch(
            texts,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Build output
        result = []
        for sub, translated_text in zip(subtitles, translations):
            new_sub = sub.copy()
            new_sub['original_text'] = sub['text']
            new_sub['text'] = translated_text
            result.append(new_sub)
        
        return result
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_path: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> 'NMTTranslator':
        """Load translator from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint.
            tokenizer_path: Path to tokenizer model file.
            device: Device for inference.
            **kwargs: Additional arguments passed to __init__.
        
        Returns:
            Initialized NMTTranslator.
        """
        from ..tokenizer import Tokenizer
        from ..model.transformer import Transformer
        
        # Load tokenizer
        tokenizer = Tokenizer(tokenizer_path)
        
        # Load checkpoint
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model from config
        config = checkpoint.get('config', {})
        config['vocab_size'] = tokenizer.vocab_size
        
        model = Transformer.from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, tokenizer, device=device, **kwargs)
    
    def save(self, path: str):
        """Save model and config for later loading.
        
        Args:
            path: Directory to save model and config.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.model.get_config()
        }, path / 'model.pt')
        
        print(f"Model saved to {path}")
