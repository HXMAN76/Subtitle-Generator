"""Custom translation model module (from scratch, no built-in APIs)."""

import torch
import torch.nn as nn
import json
from pathlib import Path
import config


class Seq2SeqTranslator(nn.Module):
    """Sequence-to-sequence translator using encoder-decoder architecture."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 2):
        """Initialize the translation model.
        
        Args:
            vocab_size: Size of the vocabulary.
            embedding_dim: Dimension of word embeddings.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of LSTM layers.
        """
        super(Seq2SeqTranslator, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                               batch_first=True, bidirectional=True)
        
        # Decoder
        self.decoder = nn.LSTM(embedding_dim, hidden_dim * 2, num_layers, 
                               batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, src, tgt):
        """Forward pass through the model."""
        # Encoder
        embedded_src = self.embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder(embedded_src)
        
        # Decoder
        embedded_tgt = self.embedding(tgt)
        decoder_outputs, _ = self.decoder(embedded_tgt, (hidden, cell))
        
        # Output
        output = self.fc(decoder_outputs)
        return output


class Translator:
    """Handles text translation using a custom trained model."""
    
    def __init__(self, model_path: str = None, vocab_path: str = None):
        """Initialize the translator.
        
        Args:
            model_path: Path to the trained model file.
            vocab_path: Path to the vocabulary JSON file.
        """
        self.model_path = model_path or str(config.TRANSLATION_MODEL_PATH)
        self.vocab_path = vocab_path or str(config.TRANSLATION_VOCAB_PATH)
        
        self.model = None
        self.vocab = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        if Path(self.model_path).exists() and Path(self.vocab_path).exists():
            self.load_model()
        else:
            print("Warning: Translation model not found. Please train the model first.")
    
    def load_model(self):
        """Load the trained model and vocabulary."""
        try:
            # Load vocabulary
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                self.word_to_idx = vocab_data['word_to_idx']
                self.idx_to_word = vocab_data['idx_to_word']
            
            # Load model
            checkpoint = torch.load(self.model_path, map_location='cpu')
            vocab_size = len(self.word_to_idx)
            
            self.model = Seq2SeqTranslator(
                vocab_size=vocab_size,
                embedding_dim=checkpoint.get('embedding_dim', 256),
                hidden_dim=checkpoint.get('hidden_dim', 512),
                num_layers=checkpoint.get('num_layers', 2)
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print("Translation model loaded successfully.")
        except Exception as e:
            print(f"Failed to load translation model: {e}")
            self.model = None
    
    def tokenize(self, text: str) -> list:
        """Convert text to token indices.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            List of token indices.
        """
        words = text.lower().split()
        return [self.word_to_idx.get(word, self.word_to_idx.get('<unk>', 0)) 
                for word in words]
    
    def detokenize(self, indices: list) -> str:
        """Convert token indices back to text.
        
        Args:
            indices: List of token indices.
            
        Returns:
            Decoded text string.
        """
        words = [self.idx_to_word.get(str(idx), '<unk>') for idx in indices]
        return ' '.join(words)
    
    def translate(self, text: str, source_lang: str = None, 
                  target_lang: str = None) -> str:
        """Translate text from source language to target language.
        
        Args:
            text: Text to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            
        Returns:
            Translated text string.
        """
        if self.model is None:
            # Fallback: return original text if model not loaded
            print("Warning: Translation model not loaded. Returning original text.")
            return text
        
        try:
            # Tokenize input
            src_tokens = self.tokenize(text)
            src_tensor = torch.tensor([src_tokens], dtype=torch.long)
            
            # Generate translation (simple greedy decoding)
            with torch.no_grad():
                # Start with <sos> token
                tgt_tokens = [self.word_to_idx.get('<sos>', 1)]
                max_length = 100
                
                for _ in range(max_length):
                    tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long)
                    output = self.model(src_tensor, tgt_tensor)
                    
                    # Get the last predicted token
                    next_token = output[0, -1].argmax().item()
                    tgt_tokens.append(next_token)
                    
                    # Stop if <eos> token is generated
                    if next_token == self.word_to_idx.get('<eos>', 2):
                        break
                
                # Detokenize
                translated_text = self.detokenize(tgt_tokens[1:-1])  # Remove <sos> and <eos>
                return translated_text
        
        except Exception as e:
            print(f"Translation failed: {e}")
            return text
    
    def translate_subtitles(self, subtitles: list) -> list:
        """Translate a list of subtitle entries.
        
        Args:
            subtitles: List of subtitle dictionaries with 'text' key.
            
        Returns:
            List of subtitle dictionaries with translated text.
        """
        translated = []
        
        for subtitle in subtitles:
            translated_subtitle = subtitle.copy()
            translated_subtitle['text'] = self.translate(subtitle['text'])
            translated_subtitle['original_text'] = subtitle['text']
            translated.append(translated_subtitle)
        
        return translated
