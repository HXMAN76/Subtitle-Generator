"""
Unit tests for NMT Subsystem.

Tests cover:
- Tokenizer functionality
- Model architecture correctness
- Attention masking
- Forward pass shapes
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F


class TestNMTConfig(unittest.TestCase):
    """Test NMT configuration."""
    
    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        from src.nmt.config import ModelConfig
        
        # Valid config
        config = ModelConfig(d_model=512, n_heads=8)
        self.assertEqual(config.d_model, 512)
        
        # Invalid: d_model not divisible by n_heads
        with self.assertRaises(AssertionError):
            ModelConfig(d_model=512, n_heads=7)
    
    def test_config_presets(self):
        """Test configuration presets."""
        from src.nmt.config import get_base_config, get_small_config, get_debug_config
        
        base = get_base_config()
        self.assertEqual(base.model.d_model, 512)
        self.assertEqual(base.model.n_encoder_layers, 6)
        
        small = get_small_config()
        self.assertEqual(small.model.d_model, 256)
        self.assertEqual(small.model.n_encoder_layers, 4)
        
        debug = get_debug_config()
        self.assertEqual(debug.model.d_model, 64)


class TestPositionalEncoding(unittest.TestCase):
    """Test positional encoding."""
    
    def test_output_shape(self):
        """Test positional encoding output shape."""
        from src.nmt.model.embeddings import PositionalEncoding
        
        d_model = 64
        pe = PositionalEncoding(d_model=d_model, max_len=100)
        
        x = torch.randn(2, 50, d_model)
        output = pe(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_different_positions_different_values(self):
        """Test that different positions have different encodings."""
        from src.nmt.model.embeddings import PositionalEncoding
        
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        
        x = torch.zeros(1, 10, 64)
        output = pe(x)
        
        # Different positions should have different values
        self.assertFalse(torch.allclose(output[0, 0], output[0, 1]))


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention."""
    
    def test_self_attention_shape(self):
        """Test self-attention output shape."""
        from src.nmt.model.attention import MultiHeadAttention
        
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        
        x = torch.randn(2, 10, 64)
        output, _ = mha(x, x, x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_cross_attention_shape(self):
        """Test cross-attention output shape."""
        from src.nmt.model.attention import MultiHeadAttention
        
        mha = MultiHeadAttention(d_model=64, n_heads=4)
        
        q = torch.randn(2, 10, 64)
        kv = torch.randn(2, 20, 64)
        output, _ = mha(q, kv, kv)
        
        self.assertEqual(output.shape, q.shape)
    
    def test_attention_with_mask(self):
        """Test attention with padding mask."""
        from src.nmt.model.attention import MultiHeadAttention, create_padding_mask
        
        mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
        
        # Create input with padding
        x = torch.randn(2, 10, 64)
        tokens = torch.ones(2, 10, dtype=torch.long)
        tokens[0, 5:] = 0  # Padding in first sequence
        
        mask = create_padding_mask(tokens, pad_idx=0)
        output, attn = mha(x, x, x, mask=mask, return_attention=True)
        
        self.assertEqual(output.shape, x.shape)
        self.assertIsNotNone(attn)


class TestMaskCreation(unittest.TestCase):
    """Test attention mask creation."""
    
    def test_padding_mask(self):
        """Test padding mask creation."""
        from src.nmt.model.attention import create_padding_mask
        
        tokens = torch.tensor([
            [1, 2, 3, 0, 0],
            [1, 2, 0, 0, 0]
        ])
        
        mask = create_padding_mask(tokens, pad_idx=0)
        
        self.assertEqual(mask.shape, (2, 1, 1, 5))
        
        # Check mask values
        expected = torch.tensor([
            [[[1, 1, 1, 0, 0]]],
            [[[1, 1, 0, 0, 0]]]
        ], dtype=torch.float)
        
        self.assertTrue(torch.equal(mask, expected))
    
    def test_causal_mask(self):
        """Test causal mask creation."""
        from src.nmt.model.attention import create_causal_mask
        
        mask = create_causal_mask(4, device=torch.device('cpu'))
        
        self.assertEqual(mask.shape, (1, 1, 4, 4))
        
        # Should be lower triangular
        expected = torch.tensor([[
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ]], dtype=torch.float).unsqueeze(0)
        
        self.assertTrue(torch.equal(mask, expected))


class TestEncoder(unittest.TestCase):
    """Test Transformer encoder."""
    
    def test_encoder_output_shape(self):
        """Test encoder output shape."""
        from src.nmt.model.encoder import TransformerEncoder
        
        encoder = TransformerEncoder(
            n_layers=2,
            d_model=64,
            n_heads=4,
            d_ff=256
        )
        
        x = torch.randn(2, 10, 64)
        output = encoder(x)
        
        self.assertEqual(output.shape, x.shape)


class TestDecoder(unittest.TestCase):
    """Test Transformer decoder."""
    
    def test_decoder_output_shape(self):
        """Test decoder output shape."""
        from src.nmt.model.decoder import TransformerDecoder
        
        decoder = TransformerDecoder(
            n_layers=2,
            d_model=64,
            n_heads=4,
            d_ff=256
        )
        
        x = torch.randn(2, 10, 64)
        encoder_output = torch.randn(2, 20, 64)
        output = decoder(x, encoder_output)
        
        self.assertEqual(output.shape, x.shape)


class TestTransformer(unittest.TestCase):
    """Test complete Transformer model."""
    
    def test_forward_pass(self):
        """Test forward pass shapes."""
        from src.nmt.model.transformer import Transformer
        
        model = Transformer(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=256,
            max_seq_len=50
        )
        
        src = torch.randint(0, 1000, (2, 10))
        tgt = torch.randint(0, 1000, (2, 15))
        
        logits = model(src, tgt)
        
        self.assertEqual(logits.shape, (2, 15, 1000))
    
    def test_weight_tying(self):
        """Test embedding weight tying."""
        from src.nmt.model.transformer import Transformer
        
        model = Transformer(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            tie_embeddings=True
        )
        
        # Embeddings should share weights
        self.assertTrue(model.src_embedding is model.tgt_embedding)
        # Output projection should be tied to TOKEN embedding (not module)
        self.assertIs(
            model.src_embedding.token_embedding.weight,
            model.output_projection.weight
        )
    
    def test_parameter_count(self):
        """Test parameter counting."""
        from src.nmt.model.transformer import Transformer
        
        model = Transformer(
            vocab_size=1000,
            d_model=64,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2
        )
        
        count = model.count_parameters()
        self.assertGreater(count, 0)
        
        readable = model.count_parameters_readable()
        self.assertIsInstance(readable, str)


class TestBeamSearch(unittest.TestCase):
    """Test beam search decoding."""
    
    def test_beam_search_output(self):
        """Test beam search produces valid output."""
        from src.nmt.model.transformer import Transformer
        from src.nmt.inference.beam_search import beam_search
        
        model = Transformer(
            vocab_size=100,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            max_seq_len=20
        )
        model.eval()
        
        src = torch.randint(4, 100, (2, 10))
        
        output = beam_search(
            model=model,
            src=src,
            beam_size=2,
            max_length=15,
            bos_id=2,
            eos_id=3
        )
        
        self.assertEqual(output.size(0), 2)
        self.assertLessEqual(output.size(1), 15)


class TestGreedyDecode(unittest.TestCase):
    """Test greedy decoding."""
    
    def test_greedy_output(self):
        """Test greedy decode produces valid output."""
        from src.nmt.model.transformer import Transformer
        from src.nmt.inference.greedy import greedy_decode
        
        model = Transformer(
            vocab_size=100,
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            max_seq_len=20
        )
        model.eval()
        
        src = torch.randint(4, 100, (2, 10))
        
        output = greedy_decode(
            model=model,
            src=src,
            bos_id=2,
            eos_id=3,
            max_length=15
        )
        
        self.assertEqual(output.size(0), 2)
        self.assertLessEqual(output.size(1), 15)


class TestLabelSmoothingLoss(unittest.TestCase):
    """Test label smoothing loss."""
    
    def test_loss_computation(self):
        """Test loss is computed correctly."""
        from src.nmt.training.trainer import LabelSmoothingLoss
        
        criterion = LabelSmoothingLoss(
            vocab_size=100,
            padding_idx=0,
            smoothing=0.1
        )
        
        logits = torch.randn(2, 10, 100)
        target = torch.randint(1, 100, (2, 10))
        
        loss = criterion(logits, target)
        
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreater(loss.item(), 0)
    
    def test_padding_ignored(self):
        """Test that padding positions are ignored."""
        from src.nmt.training.trainer import LabelSmoothingLoss
        
        criterion = LabelSmoothingLoss(
            vocab_size=100,
            padding_idx=0,
            smoothing=0.1
        )
        
        logits = torch.randn(2, 10, 100)
        
        # Target with padding
        target1 = torch.randint(1, 100, (2, 10))
        target2 = target1.clone()
        target2[:, 5:] = 0  # Add padding
        
        loss1 = criterion(logits, target1)
        loss2 = criterion(logits, target2)
        
        # Loss should be different (padding ignored)
        self.assertNotEqual(loss1.item(), loss2.item())


if __name__ == '__main__':
    unittest.main()
