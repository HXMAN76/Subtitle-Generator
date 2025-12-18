"""Training script for the custom translation model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.translator import Seq2SeqTranslator
import config


class TranslationDataset(Dataset):
    """Dataset for translation training."""
    
    def __init__(self, data_path: str, word_to_idx: dict, max_length: int = 50):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the JSON data file with parallel sentences.
            word_to_idx: Word to index mapping.
            max_length: Maximum sequence length.
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.word_to_idx = word_to_idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def tokenize(self, text: str) -> list:
        """Tokenize text to indices."""
        words = text.lower().split()
        tokens = [self.word_to_idx.get(word, self.word_to_idx['<unk>']) 
                 for word in words[:self.max_length - 2]]
        
        # Add <sos> and <eos> tokens
        tokens = [self.word_to_idx['<sos>']] + tokens + [self.word_to_idx['<eos>']]
        
        # Pad to max_length
        tokens += [self.word_to_idx['<pad>']] * (self.max_length - len(tokens))
        
        return tokens[:self.max_length]
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src = torch.tensor(self.tokenize(item['source']), dtype=torch.long)
        tgt = torch.tensor(self.tokenize(item['target']), dtype=torch.long)
        
        return src, tgt


def build_vocabulary(data_path: str, min_freq: int = 2) -> tuple:
    """Build vocabulary from training data.
    
    Args:
        data_path: Path to the training data JSON file.
        min_freq: Minimum word frequency to include in vocabulary.
        
    Returns:
        Tuple of (word_to_idx, idx_to_word) dictionaries.
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    word_freq = {}
    
    for item in data:
        for text in [item['source'], item['target']]:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Create vocabulary with special tokens
    word_to_idx = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3
    }
    
    for word, freq in word_freq.items():
        if freq >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    
    idx_to_word = {str(idx): word for word, idx in word_to_idx.items()}
    
    return word_to_idx, idx_to_word


def train_model(train_data_path: str, epochs: int = 10, batch_size: int = 32,
                learning_rate: float = 0.001, embedding_dim: int = 256,
                hidden_dim: int = 512, num_layers: int = 2):
    """Train the translation model.
    
    Args:
        train_data_path: Path to training data JSON file.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        embedding_dim: Dimension of embeddings.
        hidden_dim: Dimension of hidden layers.
        num_layers: Number of LSTM layers.
    """
    print("Building vocabulary...")
    word_to_idx, idx_to_word = build_vocabulary(train_data_path)
    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    
    # Save vocabulary
    vocab_data = {
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word
    }
    
    config.TRANSLATION_VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.TRANSLATION_VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    print("Creating dataset...")
    dataset = TranslationDataset(train_data_path, word_to_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("Initializing model...")
    model = Seq2SeqTranslator(vocab_size, embedding_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass (teacher forcing)
            output = model(src, tgt[:, :-1])
            
            # Calculate loss
            output = output.reshape(-1, vocab_size)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f}")
    
    # Save model
    print("Saving model...")
    config.TRANSLATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'vocab_size': vocab_size
    }, config.TRANSLATION_MODEL_PATH)
    
    print(f"Model saved to {config.TRANSLATION_MODEL_PATH}")
    print(f"Vocabulary saved to {config.TRANSLATION_VOCAB_PATH}")


if __name__ == "__main__":
    # Example usage
    # Your training data should be in JSON format:
    # [
    #   {"source": "Hello world", "target": "Hola mundo"},
    #   {"source": "Good morning", "target": "Buenos días"},
    #   ...
    # ]
    
    train_data_path = config.DATA_DIR / "processed" / "train_data.json"
    
    if not train_data_path.exists():
        print(f"Error: Training data not found at {train_data_path}")
        print("\nTo train the model, create a JSON file with parallel sentences:")
        print('[\n  {"source": "Hello", "target": "Hola"},')
        print('  {"source": "Goodbye", "target": "Adiós"},')
        print('  ...\n]')
    else:
        train_model(
            train_data_path=str(train_data_path),
            epochs=10,
            batch_size=32,
            learning_rate=0.001
        )
