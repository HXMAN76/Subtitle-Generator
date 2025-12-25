"""
NMT Evaluation Metrics.

Implements:
- BLEU (using SacreBLEU for reproducibility)
- METEOR (captures synonyms and paraphrases)
- COMET (neural metric with best human correlation)

Important notes on metrics:
- Always report DETOKENIZED BLEU (not tokenized)
- Use SacreBLEU for standardized, reproducible scores
- COMET correlates better with human judgment than BLEU
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import warnings


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    bleu: Optional[float] = None
    bleu_signature: Optional[str] = None
    meteor: Optional[float] = None
    comet: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'bleu': self.bleu,
            'bleu_signature': self.bleu_signature,
            'meteor': self.meteor,
            'comet': self.comet
        }
    
    def __str__(self) -> str:
        parts = []
        if self.bleu is not None:
            parts.append(f"BLEU: {self.bleu:.2f}")
        if self.meteor is not None:
            parts.append(f"METEOR: {self.meteor:.4f}")
        if self.comet is not None:
            parts.append(f"COMET: {self.comet:.4f}")
        return " | ".join(parts)


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    lowercase: bool = False
) -> Dict[str, Any]:
    """Compute BLEU score using SacreBLEU.
    
    SacreBLEU (Post, 2018) provides standardized, reproducible BLEU scores.
    Always use this instead of custom BLEU implementations.
    
    Args:
        hypotheses: List of generated translations.
        references: List of reference translations.
        lowercase: Whether to lowercase before scoring.
    
    Returns:
        Dictionary with 'score' and 'signature' keys.
    
    Note:
        Report the signature for reproducibility:
        BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.2.x.x
    """
    try:
        import sacrebleu
    except ImportError:
        warnings.warn(
            "sacrebleu not installed. Install with: pip install sacrebleu"
        )
        return {'score': None, 'signature': None}
    
    # SacreBLEU expects references as List[List[str]]
    refs = [[ref] for ref in references]
    
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        refs,
        lowercase=lowercase
    )
    
    # Version-agnostic signature extraction (sacrebleu <2.0 vs >=2.0)
    signature = None
    if hasattr(bleu, 'signature'):
        signature = bleu.signature
    elif hasattr(bleu, 'get_signature'):
        signature = bleu.get_signature()
    
    return {
        'score': bleu.score,
        'signature': signature,
        'precisions': bleu.precisions,
        'bp': bleu.bp,  # Brevity penalty
        'ratio': bleu.sys_len / bleu.ref_len if bleu.ref_len > 0 else 0
    }


def compute_bleu_sentence(
    hypothesis: str,
    reference: str
) -> float:
    """Compute sentence-level BLEU score.
    
    Note: Sentence BLEU is less reliable than corpus BLEU.
    Use primarily for debugging/analysis, not for model selection.
    
    Args:
        hypothesis: Single generated translation.
        reference: Single reference translation.
    
    Returns:
        Sentence-level BLEU score.
    """
    try:
        import sacrebleu
    except ImportError:
        return 0.0
    
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
    return bleu.score


def compute_meteor(
    hypotheses: List[str],
    references: List[str]
) -> float:
    """Compute METEOR score.
    
    METEOR captures:
    - Synonym matching
    - Stemming
    - Paraphrase matching
    
    Better than BLEU for morphologically rich languages (like Hindi).
    
    Args:
        hypotheses: List of generated translations.
        references: List of reference translations.
    
    Returns:
        Corpus-level METEOR score.
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        import nltk
        
        # Ensure required NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            
    except ImportError:
        warnings.warn(
            "nltk not installed. Install with: pip install nltk"
        )
        return 0.0
    
    scores = []
    for hyp, ref in zip(hypotheses, references):
        # Tokenize
        hyp_tokens = word_tokenize(hyp.lower())
        ref_tokens = word_tokenize(ref.lower())
        
        # METEOR expects reference as list of lists
        score = meteor_score([ref_tokens], hyp_tokens)
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0.0


def compute_comet(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int = 0
) -> float:
    """Compute COMET score.
    
    COMET (Crosslingual Optimized Metric for Evaluation of Translation)
    is a neural metric trained on human judgments from WMT campaigns.
    
    It correlates better with human evaluation than BLEU/METEOR.
    
    Key insight: COMET uses source + hypothesis + reference triplets,
    making it more accurate than metrics that ignore the source.
    
    Args:
        sources: List of source texts.
        hypotheses: List of generated translations.
        references: List of reference translations.
        model_name: COMET model to use.
        batch_size: Batch size for evaluation.
        gpus: Number of GPUs (0 for CPU).
    
    Returns:
        Corpus-level COMET score.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        warnings.warn(
            "COMET not installed. Install with: pip install unbabel-comet"
        )
        return 0.0
    
    try:
        # Download model if needed (cached after first download)
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
        
        # Prepare data
        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]
        
        # Compute scores
        output = model.predict(data, batch_size=batch_size, gpus=gpus)
        
        return output.system_score
        
    except Exception as e:
        warnings.warn(f"COMET evaluation failed: {e}")
        return 0.0


def compute_ter(
    hypotheses: List[str],
    references: List[str]
) -> float:
    """Compute Translation Edit Rate (TER).
    
    TER measures the number of edits needed to transform
    the hypothesis into the reference. Lower is better.
    
    Args:
        hypotheses: List of generated translations.
        references: List of reference translations.
    
    Returns:
        Corpus-level TER score.
    """
    try:
        import sacrebleu
    except ImportError:
        return 0.0
    
    refs = [[ref] for ref in references]
    ter = sacrebleu.corpus_ter(hypotheses, refs)
    
    return ter.score


def compute_chrf(
    hypotheses: List[str],
    references: List[str]
) -> float:
    """Compute chrF score.
    
    Character n-gram F-score, useful for morphologically rich languages.
    
    Args:
        hypotheses: List of generated translations.
        references: List of reference translations.
    
    Returns:
        Corpus-level chrF score.
    """
    try:
        import sacrebleu
    except ImportError:
        return 0.0
    
    refs = [[ref] for ref in references]
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    
    return chrf.score


def evaluate_all(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    include_comet: bool = True
) -> MetricsResult:
    """Compute all available metrics.
    
    Args:
        sources: List of source texts.
        hypotheses: List of generated translations.
        references: List of reference translations.
        include_comet: Whether to compute COMET (slow, requires GPU ideally).
    
    Returns:
        MetricsResult with all computed metrics.
    """
    result = MetricsResult()
    
    # BLEU (always compute)
    bleu_result = compute_bleu(hypotheses, references)
    result.bleu = bleu_result['score']
    result.bleu_signature = bleu_result['signature']
    
    # METEOR
    result.meteor = compute_meteor(hypotheses, references)
    
    # COMET (optional, slow)
    if include_comet:
        result.comet = compute_comet(sources, hypotheses, references)
    
    return result
