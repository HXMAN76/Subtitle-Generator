"""
Evaluation Pipeline for NMT.

Provides end-to-end evaluation on test sets:
- Load test data
- Generate translations
- Compute metrics
- Generate reports
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import torch

from .metrics import evaluate_all, compute_bleu, MetricsResult


class Evaluator:
    """Evaluation pipeline for NMT models.
    
    Args:
        translator: NMTTranslator instance.
        source_lang: Source language tag.
        target_lang: Target language tag.
    """
    
    def __init__(
        self,
        translator,
        source_lang: str = "<en>",
        target_lang: str = "<hi>"
    ):
        self.translator = translator
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def evaluate_file(
        self,
        test_file: str,
        batch_size: int = 32,
        output_file: Optional[str] = None,
        include_comet: bool = False,
        verbose: bool = True
    ) -> MetricsResult:
        """Evaluate model on a test file.
        
        Args:
            test_file: Path to test JSON file.
            batch_size: Batch size for translation.
            output_file: Optional path to save translations.
            include_comet: Whether to compute COMET score.
            verbose: Whether to print progress.
        
        Returns:
            MetricsResult with all metrics.
        """
        # Load test data
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sources = [item['source'] for item in data]
        references = [item['target'] for item in data]
        
        if verbose:
            print(f"Evaluating on {len(sources)} examples...")
        
        # Generate translations in batches
        hypotheses = []
        
        for i in tqdm(range(0, len(sources), batch_size), 
                     desc="Translating", disable=not verbose):
            batch = sources[i:i + batch_size]
            translations = self.translator.translate_batch(
                batch,
                source_lang=self.source_lang,
                target_lang=self.target_lang
            )
            hypotheses.extend(translations)
        
        # Compute metrics
        result = evaluate_all(
            sources=sources,
            hypotheses=hypotheses,
            references=references,
            include_comet=include_comet
        )
        
        if verbose:
            print(f"\n{result}")
        
        # Save translations if requested
        if output_file:
            self._save_translations(
                sources, hypotheses, references, result, output_file
            )
        
        return result
    
    def evaluate_examples(
        self,
        sources: List[str],
        references: List[str],
        include_comet: bool = False
    ) -> MetricsResult:
        """Evaluate on a list of examples.
        
        Args:
            sources: List of source texts.
            references: List of reference translations.
            include_comet: Whether to compute COMET.
        
        Returns:
            MetricsResult with all metrics.
        """
        hypotheses = self.translator.translate_batch(
            sources,
            source_lang=self.source_lang,
            target_lang=self.target_lang
        )
        
        return evaluate_all(
            sources=sources,
            hypotheses=hypotheses,
            references=references,
            include_comet=include_comet
        )
    
    def sample_translations(
        self,
        test_file: str,
        n_samples: int = 10,
        seed: int = 42
    ) -> List[Dict[str, str]]:
        """Get sample translations for qualitative analysis.
        
        Args:
            test_file: Path to test file.
            n_samples: Number of samples to show.
            seed: Random seed for sampling.
        
        Returns:
            List of dicts with source, reference, and hypothesis.
        """
        import random
        random.seed(seed)
        
        # Load test data
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Sample
        samples = random.sample(data, min(n_samples, len(data)))
        
        sources = [s['source'] for s in samples]
        references = [s['target'] for s in samples]
        
        hypotheses = self.translator.translate_batch(
            sources,
            source_lang=self.source_lang,
            target_lang=self.target_lang
        )
        
        results = []
        for src, ref, hyp in zip(sources, references, hypotheses):
            results.append({
                'source': src,
                'reference': ref,
                'hypothesis': hyp
            })
        
        return results
    
    def _save_translations(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        metrics: MetricsResult,
        output_file: str
    ):
        """Save translations and metrics to file."""
        output = {
            'metrics': metrics.to_dict(),
            'translations': [
                {
                    'source': src,
                    'reference': ref,
                    'hypothesis': hyp
                }
                for src, ref, hyp in zip(sources, references, hypotheses)
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"Translations saved to {output_file}")
    
    @staticmethod
    def compare_models(
        test_file: str,
        translators: Dict[str, Any],
        source_lang: str = "<en>",
        target_lang: str = "<hi>",
        batch_size: int = 32
    ) -> Dict[str, MetricsResult]:
        """Compare multiple models on the same test set.
        
        Args:
            test_file: Path to test file.
            translators: Dict of name -> translator mappings.
            source_lang: Source language tag.
            target_lang: Target language tag.
            batch_size: Batch size for translation.
        
        Returns:
            Dict of name -> MetricsResult mappings.
        """
        results = {}
        
        for name, translator in translators.items():
            print(f"\nEvaluating {name}...")
            evaluator = Evaluator(translator, source_lang, target_lang)
            result = evaluator.evaluate_file(
                test_file, 
                batch_size=batch_size,
                verbose=False
            )
            results[name] = result
            print(f"  {result}")
        
        return results


def quick_evaluate(
    model,
    tokenizer,
    test_file: str,
    source_lang: str = "<en>",
    target_lang: str = "<hi>",
    beam_size: int = 4,
    device: Optional[torch.device] = None
) -> MetricsResult:
    """Quick evaluation function.
    
    Convenience function for evaluating a model on a test file.
    
    Args:
        model: Trained Transformer model.
        tokenizer: Tokenizer instance.
        test_file: Path to test JSON file.
        source_lang: Source language tag.
        target_lang: Target language tag.
        beam_size: Beam size for decoding.
        device: Device for inference.
    
    Returns:
        MetricsResult with metrics.
    """
    from ..inference import NMTTranslator
    
    translator = NMTTranslator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        beam_size=beam_size
    )
    
    evaluator = Evaluator(translator, source_lang, target_lang)
    return evaluator.evaluate_file(test_file, include_comet=False)
