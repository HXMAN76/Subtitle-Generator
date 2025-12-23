"""
SentencePiece Tokenizer Wrapper for NMT.

Provides a clean interface for tokenization with:
- Language tag support for multilingual readiness
- Proper handling of special tokens (pad, unk, bos, eos)
- Batch encoding/decoding with padding
- Training utility for new tokenizers
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Tuple
import sentencepiece as spm


class Tokenizer:
    """SentencePiece tokenizer with multilingual support.
    
    This tokenizer wraps SentencePiece with additional features:
    - Explicit PAD token handling (SentencePiece default lacks PAD)
    - Language tags for multilingual translation
    - Batch encoding with dynamic padding
    
    Special Token IDs (after training):
        0: <pad>
        1: <unk>
        2: <bos>
        3: <eos>
        4: <en>
        5: <hi>
        ...additional language tags
    """
    
    # Default special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        language_tags: Optional[List[str]] = None
    ):
        """Initialize tokenizer.
        
        Args:
            model_path: Path to SentencePiece .model file.
                       If None, tokenizer must be trained before use.
            language_tags: List of language tags (e.g., ["<en>", "<hi>"]).
                          Used for validation after loading.
        """
        self.sp = spm.SentencePieceProcessor()
        self.model_path = model_path
        self.language_tags = language_tags or ["<en>", "<hi>"]
        
        # Special token IDs (set after loading/training)
        self.pad_id: int = 0
        self.unk_id: int = 1
        self.bos_id: int = 2
        self.eos_id: int = 3
        
        # Language tag IDs (set after loading)
        self.lang_to_id: dict = {}
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def load(self, model_path: str) -> None:
        """Load a trained SentencePiece model.
        
        Args:
            model_path: Path to the .model file.
        """
        self.model_path = model_path
        self.sp.Load(model_path)
        
        # Verify and cache special token IDs
        self.pad_id = self.sp.PieceToId(self.PAD_TOKEN)
        self.unk_id = self.sp.PieceToId(self.UNK_TOKEN)
        self.bos_id = self.sp.PieceToId(self.BOS_TOKEN)
        self.eos_id = self.sp.PieceToId(self.EOS_TOKEN)
        
        # Verify special tokens exist (not mapped to UNK)
        assert self.pad_id != self.unk_id, \
            f"PAD token not found in vocabulary. Retrain tokenizer with user_defined_symbols."
        assert self.bos_id != self.unk_id, \
            f"BOS token not found in vocabulary."
        assert self.eos_id != self.unk_id, \
            f"EOS token not found in vocabulary."
        
        # Cache language tag IDs
        for tag in self.language_tags:
            tag_id = self.sp.PieceToId(tag)
            if tag_id != self.unk_id:
                self.lang_to_id[tag] = tag_id
            else:
                print(f"Warning: Language tag '{tag}' not found in vocabulary.")
        
        print(f"Tokenizer loaded: vocab_size={self.vocab_size}, "
              f"pad_id={self.pad_id}, bos_id={self.bos_id}, eos_id={self.eos_id}")
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.sp.GetPieceSize()
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        add_lang_tag: Optional[str] = None
    ) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text string.
            add_bos: Whether to prepend BOS token.
            add_eos: Whether to append EOS token.
            add_lang_tag: Language tag to prepend (e.g., "<en>").
                         Added after BOS if add_bos=True.
        
        Returns:
            List of token IDs.
        """
        ids = self.sp.EncodeAsIds(text)
        
        # Prepend language tag if specified
        if add_lang_tag and add_lang_tag in self.lang_to_id:
            ids = [self.lang_to_id[add_lang_tag]] + ids
        
        # Add BOS/EOS
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        return ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text.
        
        Args:
            ids: List of token IDs.
            skip_special_tokens: Whether to remove special tokens from output.
        
        Returns:
            Decoded text string.
        """
        if skip_special_tokens:
            # Remove special tokens
            special_ids = {self.pad_id, self.bos_id, self.eos_id}
            special_ids.update(self.lang_to_id.values())
            ids = [i for i in ids if i not in special_ids]
        
        return self.sp.DecodeIds(ids)
    
    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
        add_lang_tag: Optional[str] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """Batch encode texts with padding.
        
        Args:
            texts: List of input texts.
            max_length: Maximum sequence length. If None, use longest in batch.
            add_bos: Whether to prepend BOS token.
            add_eos: Whether to append EOS token.
            add_lang_tag: Language tag to prepend.
            padding: Whether to pad sequences to max_length.
            truncation: Whether to truncate sequences exceeding max_length.
        
        Returns:
            Tuple of (token_ids, attention_masks).
            - token_ids: List of token ID sequences (padded).
            - attention_masks: List of attention masks (1=real, 0=padding).
        """
        # Encode all texts
        encoded = [
            self.encode(text, add_bos=add_bos, add_eos=add_eos, add_lang_tag=add_lang_tag)
            for text in texts
        ]
        
        # Determine max length
        if max_length is None:
            max_length = max(len(seq) for seq in encoded)
        
        # Truncate if needed
        if truncation:
            encoded = [seq[:max_length] for seq in encoded]
        
        # Pad sequences
        token_ids = []
        attention_masks = []
        
        for seq in encoded:
            seq_len = len(seq)
            if padding and seq_len < max_length:
                pad_len = max_length - seq_len
                padded_seq = seq + [self.pad_id] * pad_len
                mask = [1] * seq_len + [0] * pad_len
            else:
                padded_seq = seq
                mask = [1] * seq_len
            
            token_ids.append(padded_seq)
            attention_masks.append(mask)
        
        return token_ids, attention_masks
    
    def decode_batch(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Batch decode token IDs to texts.
        
        Args:
            batch_ids: List of token ID sequences.
            skip_special_tokens: Whether to remove special tokens.
        
        Returns:
            List of decoded text strings.
        """
        return [
            self.decode(ids, skip_special_tokens=skip_special_tokens)
            for ids in batch_ids
        ]
    
    def get_lang_id(self, lang: str) -> int:
        """Get token ID for a language tag.
        
        Args:
            lang: Language code (e.g., "en", "hi") or tag ("<en>", "<hi>").
        
        Returns:
            Token ID for the language tag, or UNK ID if not found.
        """
        # Normalize to tag format
        if not lang.startswith("<"):
            lang = f"<{lang}>"
        
        return self.lang_to_id.get(lang, self.unk_id)
    
    @staticmethod
    def train(
        corpus_path: str,
        model_prefix: str,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        language_tags: Optional[List[str]] = None,
        num_threads: int = 4
    ) -> "Tokenizer":
        """Train a new SentencePiece tokenizer.
        
        This method trains a BPE tokenizer with proper special token handling
        for NMT. Special tokens are defined as user_defined_symbols to ensure
        they are not split and have fixed IDs.
        
        Args:
            corpus_path: Path to training corpus (one sentence per line).
            model_prefix: Output path prefix for .model and .vocab files.
            vocab_size: Target vocabulary size.
            model_type: "bpe" or "unigram".
            character_coverage: Character coverage (higher for non-Latin scripts).
            language_tags: List of language tags (e.g., ["<en>", "<hi>"]).
            num_threads: Number of training threads.
        
        Returns:
            Trained Tokenizer instance.
        """
        language_tags = language_tags or ["<en>", "<hi>"]
        
        # Create output directory if needed
        os.makedirs(Path(model_prefix).parent, exist_ok=True)
        
        # Define user symbols: PAD, UNK, BOS, EOS, then language tags
        # Order matters! These get IDs 0, 1, 2, 3, 4, 5, ...
        user_defined_symbols = [
            Tokenizer.PAD_TOKEN,
            Tokenizer.UNK_TOKEN,
            Tokenizer.BOS_TOKEN,
            Tokenizer.EOS_TOKEN,
        ] + language_tags
        
        # Build training command
        # Note: We use control_symbols for tokens that shouldn't be sampled,
        # and user_defined_symbols for tokens that should be preserved as-is
        train_args = [
            f"--input={corpus_path}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            f"--model_type={model_type}",
            f"--character_coverage={character_coverage}",
            f"--num_threads={num_threads}",
            # Pad handling
            f"--pad_id=0",
            f"--pad_piece={Tokenizer.PAD_TOKEN}",
            # UNK handling
            f"--unk_id=1",
            f"--unk_piece={Tokenizer.UNK_TOKEN}",
            # BOS handling
            f"--bos_id=2",
            f"--bos_piece={Tokenizer.BOS_TOKEN}",
            # EOS handling
            f"--eos_id=3",
            f"--eos_piece={Tokenizer.EOS_TOKEN}",
            # Additional user-defined symbols (language tags)
            f"--user_defined_symbols={','.join(language_tags)}",
            # Normalization (preserve original text as much as possible)
            "--normalization_rule_name=identity",
            # Split digits for better number handling
            "--split_digits=true",
            # Byte fallback for unknown characters
            "--byte_fallback=true",
            # Vocabulary control
            "--max_sentence_length=16384",
            "--shuffle_input_sentence=true",
        ]
        
        print(f"Training SentencePiece tokenizer...")
        print(f"  Corpus: {corpus_path}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Model type: {model_type}")
        print(f"  Language tags: {language_tags}")
        
        spm.SentencePieceTrainer.Train(" ".join(train_args))
        
        print(f"Tokenizer trained: {model_prefix}.model")
        
        # Load and return the trained tokenizer
        tokenizer = Tokenizer(
            model_path=f"{model_prefix}.model",
            language_tags=language_tags
        )
        
        return tokenizer


def train_nmt_tokenizer(
    corpus_path: str = "data/raw/spm_corpus.txt",
    output_prefix: str = "models/translation/nmt_spm",
    vocab_size: int = 32000,
    language_tags: List[str] = None
) -> Tokenizer:
    """Convenience function to train the NMT tokenizer.
    
    Args:
        corpus_path: Path to corpus file.
        output_prefix: Output path prefix.
        vocab_size: Vocabulary size.
        language_tags: Language tags to include.
    
    Returns:
        Trained Tokenizer instance.
    """
    if language_tags is None:
        language_tags = ["<en>", "<hi>"]
    
    return Tokenizer.train(
        corpus_path=corpus_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,  # High coverage for Hindi
        language_tags=language_tags
    )


if __name__ == "__main__":
    # Train a new tokenizer when run directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NMT tokenizer")
    parser.add_argument("--corpus", default="data/raw/spm_corpus.txt",
                       help="Path to training corpus")
    parser.add_argument("--output", default="models/translation/nmt_spm",
                       help="Output prefix for model files")
    parser.add_argument("--vocab-size", type=int, default=32000,
                       help="Vocabulary size")
    parser.add_argument("--langs", nargs="+", default=["<en>", "<hi>"],
                       help="Language tags")
    
    args = parser.parse_args()
    
    tokenizer = train_nmt_tokenizer(
        corpus_path=args.corpus,
        output_prefix=args.output,
        vocab_size=args.vocab_size,
        language_tags=args.langs
    )
    
    # Test the tokenizer
    print("\n--- Tokenizer Test ---")
    test_en = "Hello, how are you today?"
    test_hi = "आज आप कैसे हैं?"
    
    print(f"\nEnglish: {test_en}")
    ids = tokenizer.encode(test_en, add_lang_tag="<en>")
    print(f"  Encoded: {ids[:20]}...")
    print(f"  Decoded: {tokenizer.decode(ids)}")
    
    print(f"\nHindi: {test_hi}")
    ids = tokenizer.encode(test_hi, add_lang_tag="<hi>")
    print(f"  Encoded: {ids[:20]}...")
    print(f"  Decoded: {tokenizer.decode(ids)}")
