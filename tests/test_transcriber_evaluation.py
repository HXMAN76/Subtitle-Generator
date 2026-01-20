"""
Unit tests for Transcriber Evaluation Module.

Tests cover:
- WER (Word Error Rate) computation
- CER (Character Error Rate) computation
- SER (Sentence Error Rate) computation
- Levenshtein distance algorithm
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcriber_evaluation import (
    compute_wer,
    compute_cer,
    compute_ser,
    evaluate_transcriptions,
    levenshtein_distance
)


class TestLevenshteinDistance(unittest.TestCase):
    """Test Levenshtein distance computation."""
    
    def test_identical_sequences(self):
        """Test distance between identical sequences."""
        ref = ["hello", "world"]
        hyp = ["hello", "world"]
        dist, subs, dels, ins = levenshtein_distance(ref, hyp)
        
        self.assertEqual(dist, 0)
        self.assertEqual(subs, 0)
        self.assertEqual(dels, 0)
        self.assertEqual(ins, 0)
    
    def test_substitution(self):
        """Test substitution operation."""
        ref = ["hello", "world"]
        hyp = ["hello", "earth"]
        dist, subs, dels, ins = levenshtein_distance(ref, hyp)
        
        self.assertEqual(dist, 1)
        self.assertEqual(subs, 1)
        self.assertEqual(dels, 0)
        self.assertEqual(ins, 0)
    
    def test_deletion(self):
        """Test deletion operation."""
        ref = ["hello", "beautiful", "world"]
        hyp = ["hello", "world"]
        dist, subs, dels, ins = levenshtein_distance(ref, hyp)
        
        self.assertEqual(dist, 1)
        self.assertEqual(subs, 0)
        self.assertEqual(dels, 1)
        self.assertEqual(ins, 0)
    
    def test_insertion(self):
        """Test insertion operation."""
        ref = ["hello", "world"]
        hyp = ["hello", "beautiful", "world"]
        dist, subs, dels, ins = levenshtein_distance(ref, hyp)
        
        self.assertEqual(dist, 1)
        self.assertEqual(subs, 0)
        self.assertEqual(dels, 0)
        self.assertEqual(ins, 1)
    
    def test_multiple_operations(self):
        """Test mix of operations."""
        ref = ["the", "quick", "brown", "fox"]
        hyp = ["a", "fast", "brown", "dog"]
        dist, subs, dels, ins = levenshtein_distance(ref, hyp)
        
        # "the" -> "a" (sub), "quick" -> "fast" (sub), "fox" -> "dog" (sub)
        self.assertEqual(dist, 3)
        self.assertEqual(subs, 3)


class TestWER(unittest.TestCase):
    """Test Word Error Rate computation."""
    
    def test_perfect_match(self):
        """Test WER with perfect transcription."""
        refs = ["hello world"]
        hyps = ["hello world"]
        
        result = compute_wer(refs, hyps)
        
        self.assertEqual(result['wer'], 0.0)
        self.assertEqual(result['total_words'], 2)
    
    def test_single_substitution(self):
        """Test WER with one substitution."""
        refs = ["hello world"]
        hyps = ["hello earth"]
        
        result = compute_wer(refs, hyps)
        
        # 1 substitution out of 2 words = 0.5
        self.assertEqual(result['wer'], 0.5)
        self.assertEqual(result['substitutions'], 1)
        self.assertEqual(result['deletions'], 0)
        self.assertEqual(result['insertions'], 0)
    
    def test_deletion(self):
        """Test WER with deletions."""
        refs = ["the quick brown fox"]
        hyps = ["the brown fox"]
        
        result = compute_wer(refs, hyps)
        
        # 1 deletion out of 4 words = 0.25
        self.assertEqual(result['wer'], 0.25)
        self.assertEqual(result['deletions'], 1)
    
    def test_insertion(self):
        """Test WER with insertions."""
        refs = ["hello world"]
        hyps = ["hello beautiful world"]
        
        result = compute_wer(refs, hyps)
        
        # 1 insertion out of 2 words = 0.5
        self.assertEqual(result['wer'], 0.5)
        self.assertEqual(result['insertions'], 1)
    
    def test_normalization(self):
        """Test text normalization."""
        refs = ["Hello World"]
        hyps = ["hello world"]
        
        result = compute_wer(refs, hyps, normalize=True)
        self.assertEqual(result['wer'], 0.0)
        
        result = compute_wer(refs, hyps, normalize=False)
        self.assertEqual(result['wer'], 1.0)  # Both words differ in case


class TestCER(unittest.TestCase):
    """Test Character Error Rate computation."""
    
    def test_perfect_match(self):
        """Test CER with perfect match."""
        refs = ["hello"]
        hyps = ["hello"]
        
        cer = compute_cer(refs, hyps)
        self.assertEqual(cer, 0.0)
    
    def test_character_substitution(self):
        """Test CER with character substitutions."""
        refs = ["hello"]      # 5 chars
        hyps = ["hallo"]      # 1 substitution: e -> a
        
        cer = compute_cer(refs, hyps)
        self.assertEqual(cer, 0.2)  # 1/5 = 0.2
    
    def test_character_deletion(self):
        """Test CER with character deletion."""
        refs = ["hello"]      # 5 chars
        hyps = ["helo"]       # 1 deletion: missing 'l'
        
        cer = compute_cer(refs, hyps)
        self.assertEqual(cer, 0.2)  # 1/5 = 0.2


class TestSER(unittest.TestCase):
    """Test Sentence Error Rate computation."""
    
    def test_all_correct(self):
        """Test SER with all correct sentences."""
        refs = ["hello world", "how are you"]
        hyps = ["hello world", "how are you"]
        
        ser = compute_ser(refs, hyps)
        self.assertEqual(ser, 0.0)
    
    def test_some_errors(self):
        """Test SER with some errors."""
        refs = ["hello world", "how are you", "goodbye"]
        hyps = ["hello world", "how are u", "goodbye"]
        
        ser = compute_ser(refs, hyps)
        self.assertAlmostEqual(ser, 1/3)  # 1 sentence error out of 3
    
    def test_all_errors(self):
        """Test SER with all errors."""
        refs = ["hello world", "goodbye"]
        hyps = ["hi earth", "bye"]
        
        ser = compute_ser(refs, hyps)
        self.assertEqual(ser, 1.0)


class TestEvaluateTranscriptions(unittest.TestCase):
    """Test complete evaluation pipeline."""
    
    def test_evaluate_all_metrics(self):
        """Test evaluation with all metrics."""
        refs = ["hello world", "how are you"]
        hyps = ["hello earth", "how r you"]
        
        result = evaluate_transcriptions(refs, hyps)
        
        # Check all metrics are computed
        self.assertIsNotNone(result.wer)
        self.assertIsNotNone(result.cer)
        self.assertIsNotNone(result.ser)
        
        # WER should be > 0 (we have errors)
        self.assertGreater(result.wer, 0)
        
        # Both sentences have errors
        self.assertEqual(result.ser, 1.0)
    
    def test_evaluate_with_rtf(self):
        """Test evaluation with RTF computation."""
        refs = ["hello world"]
        hyps = ["hello world"]
        audio_durations = [5.0]
        processing_times = [2.5]
        
        result = evaluate_transcriptions(
            refs, hyps,
            audio_durations=audio_durations,
            processing_times=processing_times
        )
        
        # RTF = processing_time / audio_duration = 2.5 / 5.0 = 0.5
        self.assertEqual(result.rtf, 0.5)


if __name__ == '__main__':
    unittest.main()
