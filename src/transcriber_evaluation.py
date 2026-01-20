"""
ASR (Automatic Speech Recognition) Evaluation Metrics.

Implements industry-standard metrics for evaluating transcription quality:
- WER (Word Error Rate): Primary ASR metric
- CER (Character Error Rate): Better for non-Latin scripts
- SER (Sentence Error Rate): Percentage of sentences with errors
- RTF (Real-Time Factor): Processing speed metric

Usage:
    from src.transcriber_evaluation import compute_wer, compute_cer, ASRMetricsResult
    
    result = evaluate_transcriptions(
        references=["hello world", "this is a test"],
        hypotheses=["hello world", "this is test"]
    )
    print(f"WER: {result.wer:.2%}")
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time
import warnings


@dataclass
class ASRMetricsResult:
    """Container for ASR evaluation metrics."""
    wer: Optional[float] = None  # Word Error Rate (0-1, lower is better)
    cer: Optional[float] = None  # Character Error Rate (0-1, lower is better)
    ser: Optional[float] = None  # Sentence Error Rate (0-1, lower is better)
    rtf: Optional[float] = None  # Real-Time Factor (processing_time / audio_duration)
    
    # Detailed WER statistics
    substitutions: Optional[int] = None
    deletions: Optional[int] = None
    insertions: Optional[int] = None
    total_words: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'wer': self.wer,
            'cer': self.cer,
            'ser': self.ser,
            'rtf': self.rtf,
            'substitutions': self.substitutions,
            'deletions': self.deletions,
            'insertions': self.insertions,
            'total_words': self.total_words
        }
    
    def __str__(self) -> str:
        parts = []
        if self.wer is not None:
            parts.append(f"WER: {self.wer:.2%}")
        if self.cer is not None:
            parts.append(f"CER: {self.cer:.2%}")
        if self.ser is not None:
            parts.append(f"SER: {self.ser:.2%}")
        if self.rtf is not None:
            parts.append(f"RTF: {self.rtf:.3f}")
        return " | ".join(parts)


def levenshtein_distance(ref: List[str], hyp: List[str]) -> tuple:
    """Compute Levenshtein distance with operation counts.
    
    Returns the minimum edit distance along with counts of
    substitutions, deletions, and insertions.
    
    Args:
        ref: Reference tokens (words or characters).
        hyp: Hypothesis tokens.
        
    Returns:
        Tuple of (distance, substitutions, deletions, insertions).
    """
    m, n = len(ref), len(hyp)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Operation tracking: 0=match, 1=sub, 2=del, 3=ins
    ops = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            ops[i][0] = 2  # Deletion
    
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            ops[0][j] = 3  # Insertion
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = 0  # Match
            else:
                # Substitution
                sub_cost = dp[i - 1][j - 1] + 1
                # Deletion
                del_cost = dp[i - 1][j] + 1
                # Insertion
                ins_cost = dp[i][j - 1] + 1
                
                min_cost = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = min_cost
                
                if min_cost == sub_cost:
                    ops[i][j] = 1  # Substitution
                elif min_cost == del_cost:
                    ops[i][j] = 2  # Deletion
                else:
                    ops[i][j] = 3  # Insertion
    
    # Backtrack to count operations
    i, j = m, n
    sub_count = del_count = ins_count = 0
    
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == 0:  # Match
            i -= 1
            j -= 1
        elif op == 1:  # Substitution
            sub_count += 1
            i -= 1
            j -= 1
        elif op == 2:  # Deletion
            del_count += 1
            i -= 1
        else:  # Insertion
            ins_count += 1
            j -= 1
    
    return dp[m][n], sub_count, del_count, ins_count


def compute_wer(
    references: List[str],
    hypotheses: List[str],
    normalize: bool = True
) -> Dict[str, Any]:
    """Compute Word Error Rate (WER).
    
    WER = (S + D + I) / N
    where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = number of words in reference
    
    Args:
        references: List of reference transcriptions.
        hypotheses: List of generated transcriptions.
        normalize: Whether to normalize text (lowercase, strip).
        
    Returns:
        Dictionary with 'wer', 'substitutions', 'deletions', 'insertions', 'total_words'.
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have same length")
    
    total_sub = total_del = total_ins = total_words = 0
    
    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref = ref.lower().strip()
            hyp = hyp.lower().strip()
        
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        _, subs, dels, ins = levenshtein_distance(ref_words, hyp_words)
        
        total_sub += subs
        total_del += dels
        total_ins += ins
        total_words += len(ref_words)
    
    wer = (total_sub + total_del + total_ins) / total_words if total_words > 0 else 0.0
    
    return {
        'wer': wer,
        'substitutions': total_sub,
        'deletions': total_del,
        'insertions': total_ins,
        'total_words': total_words
    }


def compute_cer(
    references: List[str],
    hypotheses: List[str],
    normalize: bool = True
) -> float:
    """Compute Character Error Rate (CER).
    
    Similar to WER but operates at character level.
    Better for languages with complex morphology or non-Latin scripts.
    
    Args:
        references: List of reference transcriptions.
        hypotheses: List of generated transcriptions.
        normalize: Whether to normalize text.
        
    Returns:
        Character error rate (0-1, lower is better).
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have same length")
    
    total_distance = 0
    total_chars = 0
    
    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref = ref.lower().strip()
            hyp = hyp.lower().strip()
        
        # Convert to character lists
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        
        distance, _, _, _ = levenshtein_distance(ref_chars, hyp_chars)
        
        total_distance += distance
        total_chars += len(ref_chars)
    
    cer = total_distance / total_chars if total_chars > 0 else 0.0
    return cer


def compute_ser(
    references: List[str],
    hypotheses: List[str],
    normalize: bool = True
) -> float:
    """Compute Sentence Error Rate (SER).
    
    Percentage of sentences that have at least one error.
    
    Args:
        references: List of reference transcriptions.
        hypotheses: List of generated transcriptions.
        normalize: Whether to normalize text.
        
    Returns:
        Sentence error rate (0-1, lower is better).
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have same length")
    
    errors = 0
    
    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref = ref.lower().strip()
            hyp = hyp.lower().strip()
        
        if ref != hyp:
            errors += 1
    
    ser = errors / len(references) if references else 0.0
    return ser


def evaluate_transcriptions(
    references: List[str],
    hypotheses: List[str],
    audio_durations: Optional[List[float]] = None,
    processing_times: Optional[List[float]] = None,
    normalize: bool = True
) -> ASRMetricsResult:
    """Evaluate transcription quality with all ASR metrics.
    
    Args:
        references: List of ground truth transcriptions.
        hypotheses: List of generated transcriptions.
        audio_durations: Optional list of audio durations in seconds.
        processing_times: Optional list of processing times in seconds.
        normalize: Whether to normalize text before comparison.
        
    Returns:
        ASRMetricsResult with all computed metrics.
    """
    result = ASRMetricsResult()
    
    # Compute WER
    wer_results = compute_wer(references, hypotheses, normalize=normalize)
    result.wer = wer_results['wer']
    result.substitutions = wer_results['substitutions']
    result.deletions = wer_results['deletions']
    result.insertions = wer_results['insertions']
    result.total_words = wer_results['total_words']
    
    # Compute CER
    result.cer = compute_cer(references, hypotheses, normalize=normalize)
    
    # Compute SER
    result.ser = compute_ser(references, hypotheses, normalize=normalize)
    
    # Compute RTF if timing info provided
    if audio_durations and processing_times:
        if len(audio_durations) != len(processing_times):
            warnings.warn("Audio durations and processing times have different lengths")
        else:
            total_audio = sum(audio_durations)
            total_processing = sum(processing_times)
            result.rtf = total_processing / total_audio if total_audio > 0 else 0.0
    
    return result


class TranscriberEvaluator:
    """Evaluator for Transcriber models.
    
    Args:
        transcriber: Transcriber instance.
        normalize: Whether to normalize text before comparison.
    """
    
    def __init__(self, transcriber, normalize: bool = True):
        self.transcriber = transcriber
        self.normalize = normalize
    
    def evaluate_dataset(
        self,
        test_data: List[Dict[str, Any]],
        verbose: bool = True
    ) -> ASRMetricsResult:
        """Evaluate on a dataset of audio files.
        
        Args:
            test_data: List of dicts with 'audio', 'reference', 'duration' keys.
            verbose: Whether to print progress.
            
        Returns:
            ASRMetricsResult with evaluation metrics.
        """
        references = []
        hypotheses = []
        audio_durations = []
        processing_times = []
        
        for i, item in enumerate(test_data):
            if verbose:
                print(f"Transcribing {i+1}/{len(test_data)}: {item['audio']}", end='\r')
            
            # Transcribe
            start_time = time.time()
            result = self.transcriber.transcribe_audio(item['audio'])
            processing_time = time.time() - start_time
            
            # Collect results
            references.append(item['reference'])
            hypotheses.append(result['text'])
            
            if 'duration' in item:
                audio_durations.append(item['duration'])
                processing_times.append(processing_time)
        
        if verbose:
            print()  # New line after progress
        
        # Compute metrics
        result = evaluate_transcriptions(
            references=references,
            hypotheses=hypotheses,
            audio_durations=audio_durations if audio_durations else None,
            processing_times=processing_times if processing_times else None,
            normalize=self.normalize
        )
        
        if verbose:
            print(f"\n{result}")
        
        return result
