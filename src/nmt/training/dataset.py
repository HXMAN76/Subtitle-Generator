"""
Translation Dataset and DataLoader utilities.

Handles:
- Loading parallel corpus from JSON
- Tokenizing source and target sequences
- Dynamic batching with padding
- Bucket sampling for efficient training
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset


class TranslationDataset(Dataset):
    """Dataset for parallel translation corpus.
    
    Loads source-target sentence pairs from JSON format:
    [
        {"source": "Hello", "target": "नमस्ते"},
        {"source": "Goodbye", "target": "अलविदा"},
        ...
    ]
    
    Args:
        data_path: Path to JSON file with parallel sentences.
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.
        source_lang: Source language tag (e.g., "<en>").
        target_lang: Target language tag (e.g., "<hi>").
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 256,
        source_lang: str = "<en>",
        target_lang: str = "<hi>"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # Load data
        self.data = self._load_data(data_path)
        
        print(f"Loaded {len(self.data)} translation pairs from {data_path}")
    
    def _load_data(self, data_path: str) -> List[Dict[str, str]]:
        """Load parallel sentences from JSON or JSONL file.
        
        Supports:
        - JSON array format: [{"source": "...", "target": "..."}, ...]
        - JSONL format: {"source": "...", "target": "..."}\n{"source": "...", "target": "..."}
        - Both {source, target} and {src, tgt} key formats (Samanantar uses src/tgt)
        """
        data = []
        path = Path(data_path)
        
        # Detect format by extension or content
        if path.suffix == '.jsonl' or str(data_path).endswith('.jsonl'):
            # JSONL format (one JSON object per line)
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        data.append(self._normalize_item(item))
        else:
            # Try JSON array format first
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            if content.startswith('['):
                # JSON array
                items = json.loads(content)
                data = [self._normalize_item(item) for item in items]
            else:
                # Actually JSONL despite extension
                for line in content.split('\n'):
                    if line.strip():
                        item = json.loads(line)
                        data.append(self._normalize_item(item))
        
        return data
    
    def _normalize_item(self, item: Dict[str, str]) -> Dict[str, str]:
        """Normalize item keys to {source, target} format.
        
        Supports both {source, target} and {src, tgt} (Samanantar format).
        """
        if 'source' in item and 'target' in item:
            return item
        elif 'src' in item and 'tgt' in item:
            return {'source': item['src'], 'target': item['tgt']}
        else:
            raise ValueError(
                f"Item must have either {{source, target}} or {{src, tgt}} keys. "
                f"Got: {list(item.keys())}"
            )
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single translation pair.
        
        Returns:
            Dictionary with:
            - src_ids: Source token IDs
            - tgt_ids: Target token IDs (input to decoder, with BOS)
            - labels: Target labels (for loss, with EOS, shifted)
        """
        item = self.data[idx]
        
        # Encode source with language tag
        src_ids = self.tokenizer.encode(
            item['source'],
            add_bos=True,
            add_eos=True,
            add_lang_tag=self.source_lang
        )
        
        # Encode target with language tag
        tgt_ids = self.tokenizer.encode(
            item['target'],
            add_bos=True,
            add_eos=True,
            add_lang_tag=self.target_lang
        )
        
        # Truncate if needed
        src_ids = src_ids[:self.max_length]
        tgt_ids = tgt_ids[:self.max_length]
        
        # For training:
        # - Decoder input: [BOS, lang, tok1, tok2, ..., tokN]
        # - Labels: [lang, tok1, tok2, ..., tokN, EOS]
        # We shift inside the model, so labels = tgt_ids[1:] + padding
        
        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids[:-1], dtype=torch.long),  # Input (no EOS)
            'labels': torch.tensor(tgt_ids[1:], dtype=torch.long),    # Labels (no BOS)
        }


class TranslationDatasetStreaming(IterableDataset):
    """Streaming dataset for very large files.
    
    Loads data lazily to handle files that don't fit in memory.
    Uses line-by-line JSON format (JSON Lines / JSONL).
    
    Args:
        data_path: Path to JSONL file.
        tokenizer: Tokenizer instance.
        max_length: Maximum sequence length.
        source_lang: Source language tag.
        target_lang: Target language tag.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 256,
        source_lang: str = "<en>",
        target_lang: str = "<hi>",
        shuffle_buffer: int = 10000
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.shuffle_buffer = shuffle_buffer
        
        # Count lines for length
        self._length = None
    
    def __len__(self) -> int:
        if self._length is None:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self._length = sum(1 for _ in f)
        return self._length
    
    def __iter__(self):
        """Iterate through dataset with optional shuffling."""
        buffer = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                buffer.append(item)
                
                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    for item in buffer:
                        yield self._process_item(item)
                    buffer = []
        
        # Process remaining items
        if buffer:
            random.shuffle(buffer)
            for item in buffer:
                yield self._process_item(item)
    
    def _normalize_item(self, item: Dict[str, str]) -> Dict[str, str]:
        """Normalize item keys to {source, target} format."""
        if 'source' in item and 'target' in item:
            return item
        elif 'src' in item and 'tgt' in item:
            return {'source': item['src'], 'target': item['tgt']}
        else:
            raise ValueError(
                f"Item must have either {{source, target}} or {{src, tgt}} keys."
            )
    
    def _process_item(self, item: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Process a single item."""
        # Normalize keys first
        item = self._normalize_item(item)
        
        src_ids = self.tokenizer.encode(
            item['source'],
            add_bos=True,
            add_eos=True,
            add_lang_tag=self.source_lang
        )[:self.max_length]
        
        tgt_ids = self.tokenizer.encode(
            item['target'],
            add_bos=True,
            add_eos=True,
            add_lang_tag=self.target_lang
        )[:self.max_length]
        
        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids[:-1], dtype=torch.long),
            'labels': torch.tensor(tgt_ids[1:], dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate batch with padding.
    
    Pads all sequences to the maximum length in the batch.
    
    Args:
        batch: List of samples from dataset.
        pad_id: Padding token ID.
    
    Returns:
        Dictionary with padded tensors.
    """
    # Get max lengths in this batch
    max_src_len = max(item['src_ids'].size(0) for item in batch)
    max_tgt_len = max(item['tgt_ids'].size(0) for item in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    src_ids = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long)
    tgt_ids = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    
    # Fill in actual values
    for i, item in enumerate(batch):
        src_len = item['src_ids'].size(0)
        tgt_len = item['tgt_ids'].size(0)
        
        src_ids[i, :src_len] = item['src_ids']
        tgt_ids[i, :tgt_len] = item['tgt_ids']
        labels[i, :tgt_len] = item['labels']
    
    return {
        'src_ids': src_ids,
        'tgt_ids': tgt_ids,
        'labels': labels,
    }


class BucketSampler(Sampler):
    """Bucket sampler for efficient batching.
    
    Groups sequences of similar lengths together to minimize padding.
    This significantly improves training efficiency.
    
    Args:
        lengths: List of sequence lengths.
        batch_size: Batch size.
        bucket_size: Number of batches per bucket.
        shuffle: Whether to shuffle within buckets.
    """
    
    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        bucket_size: int = 100,
        shuffle: bool = True
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
    
    def __iter__(self):
        # Sort indices by length
        indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        
        # Create buckets
        bucket_batch_size = self.batch_size * self.bucket_size
        buckets = [indices[i:i + bucket_batch_size] 
                   for i in range(0, len(indices), bucket_batch_size)]
        
        # Shuffle within buckets
        if self.shuffle:
            for bucket in buckets:
                random.shuffle(bucket)
        
        # Shuffle bucket order
        if self.shuffle:
            random.shuffle(buckets)
        
        # Yield batches
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]
    
    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def create_dataloader(
    dataset: TranslationDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pad_id: int = 0,
    use_bucket_sampling: bool = True
) -> DataLoader:
    """Create DataLoader for translation dataset.
    
    Args:
        dataset: TranslationDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of data loading workers.
        pad_id: Padding token ID.
        use_bucket_sampling: Whether to use bucket sampling.
    
    Returns:
        DataLoader instance.
    """
    # Custom collate function with pad_id
    def collate_wrapper(batch):
        return collate_fn(batch, pad_id=pad_id)
    
    if use_bucket_sampling and shuffle:
        # Compute sequence lengths for bucket sampling
        lengths = []
        for item in dataset.data:
            # Approximate length (exact would require tokenizing)
            src_len = len(item['source'].split())
            tgt_len = len(item['target'].split())
            lengths.append(max(src_len, tgt_len))
        
        sampler = BucketSampler(lengths, batch_size, shuffle=True)
        
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_wrapper,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_wrapper,
            num_workers=num_workers,
            pin_memory=True
        )


def create_streaming_dataloader(
    dataset: TranslationDatasetStreaming,
    batch_size: int,
    num_workers: int = 0,
    pad_id: int = 0
) -> DataLoader:
    """Create DataLoader for streaming translation dataset.
    
    Args:
        dataset: TranslationDatasetStreaming instance.
        batch_size: Batch size.
        num_workers: Number of workers (0 recommended for streaming).
        pad_id: Padding token ID.
    
    Returns:
        DataLoader instance.
    """
    def collate_wrapper(batch):
        return collate_fn(batch, pad_id=pad_id)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_wrapper,
        num_workers=num_workers,
        pin_memory=True
    )
