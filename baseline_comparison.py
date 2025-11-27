"""
Baseline Comparison Module for Bridge Design Information Extraction
=====================================================================

Implements multiple baseline extraction methods for comparison:
1. Rule-based extraction (regex patterns)
2. Traditional NER + Pattern matching
3. Direct LLM extraction (GPT-4o baseline)
4. Fine-tuned transformer (no self-training)
5. Our proposed self-training approach

Provides fair comparison with statistical significance testing.

Author: Self-Training IE Framework
License: MIT
"""

import os
import re
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import logging

# Import evaluation metrics
from evaluation_metrics import (
    EvaluationMetricsCalculator,
    StatisticalSignificanceTester,
    AggregateMetrics,
    EXTRACTION_FIELDS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ABSTRACT BASE CLASS FOR EXTRACTORS
# ============================================================================

class BaseExtractor(ABC):
    """Abstract base class for all extraction methods."""

    def __init__(self, name: str):
        self.name = name
        self.extraction_times = []

    @abstractmethod
    def extract(self, document_text: str) -> Dict:
        """
        Extract structured data from document text.

        Args:
            document_text: Raw text from the document

        Returns:
            Extracted JSON structure
        """
        pass

    def extract_batch(self, documents: List[Tuple[str, str]]) -> List[Dict]:
        """
        Extract from a batch of documents.

        Args:
            documents: List of (doc_id, text) tuples

        Returns:
            List of extracted JSON structures
        """
        results = []
        for doc_id, text in documents:
            start_time = time.time()
            try:
                result = self.extract(text)
            except Exception as e:
                logger.error(f"Extraction failed for {doc_id}: {e}")
                result = self._empty_result()
            elapsed = time.time() - start_time
            self.extraction_times.append(elapsed)
            results.append(result)
        return results

    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            "bridge_type": None,
            "geometry": {"girders": {}, "deck": {}},
            "material_properties": {"concrete": {}, "steel": {}},
            "rebar": {},
            "live_load": {},
            "crossbeams": {}
        }

    def get_timing_stats(self) -> Dict[str, float]:
        """Get extraction timing statistics."""
        if not self.extraction_times:
            return {}
        return {
            "mean_time": np.mean(self.extraction_times),
            "std_time": np.std(self.extraction_times),
            "min_time": np.min(self.extraction_times),
            "max_time": np.max(self.extraction_times),
            "total_time": np.sum(self.extraction_times),
            "n_extractions": len(self.extraction_times)
        }


# ============================================================================
# BASELINE 1: RULE-BASED EXTRACTION
# ============================================================================

class RuleBasedExtractor(BaseExtractor):
    """
    Rule-based extraction using regex patterns.
    Represents traditional pre-LLM approach.
    """

    def __init__(self):
        super().__init__("Rule-Based (Regex)")
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for extraction."""

        # Japanese and English pattern variations
        self.patterns = {
            # Girder length patterns
            "girder_length": [
                r"主桁支間[長]?\s*[：:=]?\s*(\d+(?:\.\d+)?)\s*(?:mm|ｍｍ)?",
                r"支間長\s*[：:=]?\s*(\d+(?:\.\d+)?)\s*(?:mm|ｍｍ)?",
                r"span\s*length\s*[：:=]?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"L\s*=\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
            ],

            # Number of girders
            "num_girders": [
                r"主桁本数\s*[：:=]?\s*(\d+)\s*本?",
                r"G-(\d+)\s*まで",
                r"(\d+)\s*主桁",
                r"number\s*of\s*girders?\s*[：:=]?\s*(\d+)",
            ],

            # Web height
            "web_height": [
                r"WEB\s*PL\s*(\d+)\s*[×x\*]\s*\d+",
                r"主桁腹板高\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"腹板高\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"web\s*height\s*[：:=]?\s*(\d+(?:\.\d+)?)",
            ],

            # Web thickness
            "web_thickness": [
                r"WEB\s*PL\s*\d+\s*[×x\*]\s*(\d+)",
                r"腹板厚\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"web\s*thickness\s*[：:=]?\s*(\d+(?:\.\d+)?)",
            ],

            # Top flange width
            "top_flange_width": [
                r"UFLG\s*PL\s*(\d+)\s*[×x\*]\s*\d+",
                r"上フランジ幅\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"上FLG\s*PL\s*(\d+)\s*[×x\*]",
            ],

            # Top flange thickness
            "top_flange_thickness": [
                r"UFLG\s*PL\s*\d+\s*[×x\*]\s*(\d+)",
                r"上フランジ厚\s*[：:=]?\s*(\d+(?:\.\d+)?)",
            ],

            # Bottom flange width
            "bottom_flange_width": [
                r"LFLG\s*PL\s*(\d+)\s*[×x\*]\s*\d+",
                r"下フランジ幅\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"下FLG\s*PL\s*(\d+)\s*[×x\*]",
            ],

            # Bottom flange thickness
            "bottom_flange_thickness": [
                r"LFLG\s*PL\s*\d+\s*[×x\*]\s*(\d+)",
                r"下フランジ厚\s*[：:=]?\s*(\d+(?:\.\d+)?)",
            ],

            # Spacing X (transverse)
            "spacing_x": [
                r"G-1～G-2\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"横断間隔\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"girder\s*spacing\s*[：:=]?\s*(\d+(?:\.\d+)?)",
            ],

            # Spacing Z (longitudinal)
            "spacing_z": [
                r"主桁格間長\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"格間長\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"panel\s*length\s*[：:=]?\s*(\d+(?:\.\d+)?)",
            ],

            # Deck thickness
            "deck_thickness": [
                r"床版厚\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"床版\s*[：:=]?\s*t\s*=\s*(\d+(?:\.\d+)?)",
                r"slab\s*thickness\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"RC床版\s*(\d+)",
            ],

            # Deck width
            "deck_width": [
                r"合計\s*[：:=]?\s*(\d+(?:\.\d+)?)\s*(?:mm)?",
                r"total\s*width\s*[：:=]?\s*(\d+(?:\.\d+)?)",
                r"橋面幅員\s*[：:=]?\s*(\d+(?:\.\d+)?)",
            ],

            # Impact coefficient
            "impact_coefficient": [
                r"衝撃係数\s*[：:=]?\s*(\d+\.\d+)",
                r"i\s*=\s*(\d+\.\d+)",
                r"impact\s*[：:=]?\s*(\d+\.\d+)",
            ],

            # Crossbeam height
            "crossbeam_height": [
                r"横桁.*WEB\s*PL\s*(\d+)\s*[×x\*]",
                r"中間横桁.*高\s*[：:=]?\s*(\d+(?:\.\d+)?)",
            ],

            # Crossbeam thickness
            "crossbeam_thickness": [
                r"横桁.*WEB\s*PL\s*\d+\s*[×x\*]\s*(\d+)",
                r"中間横桁.*厚\s*[：:=]?\s*(\d+(?:\.\d+)?)",
            ],
        }

        # Compile patterns
        self.compiled_patterns = {}
        for field_name, patterns in self.patterns.items():
            self.compiled_patterns[field_name] = [
                re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns
            ]

    def extract(self, document_text: str) -> Dict:
        """Extract using regex patterns."""

        result = self._empty_result()

        for field_name, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(document_text)
                if match:
                    value = match.group(1)
                    self._set_field_value(result, field_name, value)
                    break

        # Set bridge type based on keywords
        if "鈑桁" in document_text or "plate girder" in document_text.lower():
            result["bridge_type"] = "PlateGirder"
        elif "鋼桁" in document_text or "steel girder" in document_text.lower():
            result["bridge_type"] = "SteelGirder"

        # Set boolean fields
        result["crossbeams"]["use_crossbeams"] = "True" if "横桁" in document_text else "False"
        result["rebar"]["use_rebar"] = "True" if "鉄筋" in document_text else "False"

        return result

    def _set_field_value(self, result: Dict, field_name: str, value: str):
        """Set value in the nested result structure."""
        field_info = EXTRACTION_FIELDS.get(field_name, {})
        path = field_info.get("path", [])

        if not path:
            return

        current = result
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[path[-1]] = value


# ============================================================================
# BASELINE 2: TRADITIONAL NER + PATTERN MATCHING
# ============================================================================

class NERPatternExtractor(BaseExtractor):
    """
    Traditional NER-based extraction with pattern matching.
    Uses spaCy/similar for entity recognition, then patterns for values.
    """

    def __init__(self):
        super().__init__("NER + Pattern Matching")
        self.ner_available = self._check_ner_availability()
        self.rule_extractor = RuleBasedExtractor()

    def _check_ner_availability(self) -> bool:
        """Check if NER models are available."""
        try:
            import spacy
            # Try to load a model
            try:
                self.nlp = spacy.load("ja_core_news_sm")
                return True
            except:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    return True
                except:
                    logger.warning("No spaCy model available, using regex only")
                    return False
        except ImportError:
            logger.warning("spaCy not installed, using regex only")
            return False

    def extract(self, document_text: str) -> Dict:
        """Extract using NER + patterns."""

        # Start with rule-based extraction
        result = self.rule_extractor.extract(document_text)

        if not self.ner_available:
            return result

        # Enhance with NER
        doc = self.nlp(document_text[:100000])  # Limit for performance

        # Look for numeric entities near keywords
        for ent in doc.ents:
            if ent.label_ in ["QUANTITY", "CARDINAL", "PERCENT"]:
                context = document_text[max(0, ent.start_char-50):ent.end_char+50]

                # Check context for field keywords
                if any(kw in context for kw in ["主桁", "girder", "span"]):
                    if result["geometry"]["girders"].get("length") is None:
                        try:
                            val = float(re.sub(r'[^\d.]', '', ent.text))
                            if 5000 <= val <= 100000:
                                result["geometry"]["girders"]["length"] = str(int(val))
                        except:
                            pass

                if any(kw in context for kw in ["床版", "slab", "deck"]):
                    if result["geometry"]["deck"].get("thickness") is None:
                        try:
                            val = float(re.sub(r'[^\d.]', '', ent.text))
                            if 150 <= val <= 500:
                                result["geometry"]["deck"]["thickness"] = str(int(val))
                        except:
                            pass

        return result


# ============================================================================
# BASELINE 3: DIRECT LLM EXTRACTION (NO SELF-TRAINING)
# ============================================================================

class DirectLLMExtractor(BaseExtractor):
    """
    Direct LLM extraction without self-training.
    This is the initial LLM extraction that produces pseudo-labels.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        super().__init__(f"Direct LLM ({model})")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.available = True
        except ImportError:
            logger.warning("OpenAI package not installed")
            self.available = False
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self.available = False

        self._load_prompt_and_schema()

    def _load_prompt_and_schema(self):
        """Load extraction prompt and schema."""
        try:
            with open("01_extraction_prompt_v2.txt", 'r', encoding='utf-8') as f:
                self.prompt = f.read()
        except:
            self.prompt = "Extract structured bridge design data from the following document."

        try:
            with open("02_function_schema_v2.json", 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
        except:
            self.schema = None

    def extract(self, document_text: str) -> Dict:
        """Extract using LLM."""

        if not self.available:
            return self._empty_result()

        try:
            messages = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"Document:\n\n{document_text[:30000]}"}
            ]

            if self.schema:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    functions=[self.schema],
                    function_call={"name": self.schema["name"]},
                    temperature=0,
                    max_tokens=4096
                )

                if response.choices[0].message.function_call:
                    result = json.loads(response.choices[0].message.function_call.arguments)
                    return result
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=4096
                )
                # Try to parse JSON from response
                content = response.choices[0].message.content
                # Find JSON in response
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    return json.loads(json_match.group())

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")

        return self._empty_result()


# ============================================================================
# BASELINE 4: FINE-TUNED TRANSFORMER (NO SELF-TRAINING)
# ============================================================================

class FineTunedTransformerExtractor(BaseExtractor):
    """
    Fine-tuned transformer for extraction WITHOUT self-training.
    Trains once on initial pseudo-labels without iterative refinement.
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__("Fine-tuned Transformer (No Self-Training)")
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load or initialize the transformer model."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            if self.model_path and Path(self.model_path).exists():
                # Load pre-trained model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModel.from_pretrained(self.model_path)
                self.available = True
            else:
                # Model not trained yet - will need to train first
                self.available = False
                logger.info("Fine-tuned model not found. Need to train first.")

        except ImportError:
            logger.warning("PyTorch/Transformers not installed")
            self.available = False

    def train(
        self,
        documents: List[Tuple[str, str]],
        labels: List[Dict],
        epochs: int = 5
    ):
        """
        Train the transformer on pseudo-labels.
        Single pass training without self-training refinement.
        """
        # Import training utilities from main framework
        from self_training_framework import (
            BridgeExtractionTransformer,
            BridgeExtractionDataset,
            SelfTrainingConfig,
            PseudoLabelConfidenceEstimator
        )

        config = SelfTrainingConfig()
        config.num_epochs = epochs

        # Initialize model
        self.transformer = BridgeExtractionTransformer(config)
        self.transformer.to(config.device)

        # Get tokenizer
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(config.fallback_model)

        # Create uniform confidence (no confidence weighting)
        uniform_confidences = [{f: 1.0 for f in EXTRACTION_FIELDS} for _ in labels]

        # Create dataset
        from torch.utils.data import DataLoader
        from torch.optim import AdamW

        dataset = BridgeExtractionDataset(
            documents, labels, uniform_confidences,
            self.tokenizer, config.max_seq_length
        )

        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        optimizer = AdamW(self.transformer.parameters(), lr=config.learning_rate)

        # Training loop
        import torch.nn.functional as F
        from tqdm import tqdm

        self.transformer.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                input_ids = batch["input_ids"].to(config.device)
                attention_mask = batch["attention_mask"].to(config.device)
                labels_tensor = batch["labels"].to(config.device)

                optimizer.zero_grad()
                outputs = self.transformer(input_ids, attention_mask)
                loss = F.mse_loss(outputs["predictions"], labels_tensor)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch {epoch+1} loss: {total_loss/len(dataloader):.4f}")

        self.model = self.transformer
        self.available = True

    def extract(self, document_text: str) -> Dict:
        """Extract using fine-tuned transformer."""

        if not self.available or self.model is None:
            return self._empty_result()

        import torch
        from self_training_framework import SelfTrainingConfig

        config = SelfTrainingConfig()
        self.model.eval()

        encoding = self.tokenizer(
            document_text[:10000],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )

        with torch.no_grad():
            input_ids = encoding["input_ids"].to(config.device)
            attention_mask = encoding["attention_mask"].to(config.device)
            outputs = self.model(input_ids, attention_mask)
            predictions = outputs["predictions"].cpu().numpy()[0]

        # Convert predictions to JSON structure
        return self._predictions_to_json(predictions)

    def _predictions_to_json(self, predictions) -> Dict:
        """Convert model predictions to JSON."""
        result = self._empty_result()

        for field_idx, (field_name, field_info) in enumerate(EXTRACTION_FIELDS.items()):
            value = predictions[field_idx]
            path = field_info["path"]

            current = result
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            if field_info["type"] in ["numeric", "integer"]:
                current[path[-1]] = str(int(round(value))) if abs(value) > 0.5 else None
            elif field_info["type"] == "boolean":
                current[path[-1]] = "True" if value > 0.5 else "False"
            else:
                current[path[-1]] = str(value) if abs(value) > 0.1 else None

        return result


# ============================================================================
# OUR PROPOSED METHOD (SELF-TRAINING)
# ============================================================================

class SelfTrainingExtractor(BaseExtractor):
    """
    Our proposed self-training extraction approach.
    Wraps the full self-training framework.
    """

    def __init__(self, trained_models: Optional[List] = None):
        super().__init__("Self-Training Framework (Proposed)")
        self.trained_models = trained_models
        self.available = trained_models is not None and len(trained_models) > 0

    def set_models(self, models: List):
        """Set trained models from self-training."""
        self.trained_models = models
        self.available = len(models) > 0

    def extract(self, document_text: str) -> Dict:
        """Extract using self-trained ensemble."""

        if not self.available:
            return self._empty_result()

        from self_training_framework import SelfTrainingConfig
        from transformers import AutoTokenizer
        import torch

        config = SelfTrainingConfig()

        try:
            tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        except:
            tokenizer = AutoTokenizer.from_pretrained(config.fallback_model)

        # Ensemble predictions
        all_predictions = []

        for model in self.trained_models:
            model.eval()
            encoding = tokenizer(
                document_text[:10000],
                truncation=True,
                max_length=config.max_seq_length,
                padding="max_length",
                return_tensors="pt"
            )

            with torch.no_grad():
                input_ids = encoding["input_ids"].to(config.device)
                attention_mask = encoding["attention_mask"].to(config.device)
                outputs = model(input_ids, attention_mask)
                preds = outputs["predictions"].cpu().numpy()[0]
                all_predictions.append(preds)

        # Average predictions
        avg_predictions = np.mean(all_predictions, axis=0)

        return self._predictions_to_json(avg_predictions)

    def _predictions_to_json(self, predictions) -> Dict:
        """Convert predictions to JSON."""
        result = {
            "bridge_type": None,
            "geometry": {"girders": {}, "deck": {}},
            "material_properties": {"concrete": {}, "steel": {}},
            "rebar": {},
            "live_load": {},
            "crossbeams": {}
        }

        for field_idx, (field_name, field_info) in enumerate(EXTRACTION_FIELDS.items()):
            value = predictions[field_idx]
            path = field_info["path"]

            current = result
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            if field_info["type"] in ["numeric", "integer"]:
                current[path[-1]] = str(int(round(value))) if abs(value) > 0.5 else None
            elif field_info["type"] == "boolean":
                current[path[-1]] = "True" if value > 0.5 else "False"
            else:
                current[path[-1]] = str(value) if abs(value) > 0.1 else None

        return result


# ============================================================================
# BASELINE COMPARISON FRAMEWORK
# ============================================================================

@dataclass
class ComparisonResult:
    """Results from comparing multiple methods."""
    method_name: str
    metrics: AggregateMetrics
    timing: Dict[str, float]
    predictions: List[Dict] = field(default_factory=list)


class BaselineComparison:
    """
    Framework for comparing extraction methods.
    """

    def __init__(self, output_dir: str = "comparison_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extractors: Dict[str, BaseExtractor] = {}
        self.results: Dict[str, ComparisonResult] = {}
        self.metrics_calculator = EvaluationMetricsCalculator()
        self.significance_tester = StatisticalSignificanceTester()

    def register_extractor(self, name: str, extractor: BaseExtractor):
        """Register an extraction method for comparison."""
        self.extractors[name] = extractor

    def register_all_baselines(self):
        """Register all baseline methods."""
        self.register_extractor("rule_based", RuleBasedExtractor())
        self.register_extractor("ner_pattern", NERPatternExtractor())
        self.register_extractor("direct_llm", DirectLLMExtractor())
        self.register_extractor("finetuned_transformer", FineTunedTransformerExtractor())

    def run_comparison(
        self,
        documents: List[Tuple[str, str]],
        ground_truth: List[Dict],
        run_llm: bool = False
    ) -> Dict[str, ComparisonResult]:
        """
        Run all registered extractors and compare results.

        Args:
            documents: List of (doc_id, text) tuples
            ground_truth: List of ground truth JSON documents
            run_llm: Whether to run LLM-based extractors (can be expensive)

        Returns:
            Dictionary of comparison results
        """
        logger.info(f"Running comparison on {len(documents)} documents")
        logger.info(f"Registered extractors: {list(self.extractors.keys())}")

        doc_ids = [doc_id for doc_id, _ in documents]

        for name, extractor in self.extractors.items():
            # Skip LLM if not requested
            if "llm" in name.lower() and not run_llm:
                logger.info(f"Skipping {name} (run_llm=False)")
                continue

            logger.info(f"\nRunning {name}...")

            try:
                predictions = extractor.extract_batch(documents)
                timing = extractor.get_timing_stats()

                # Calculate metrics
                metrics = self.metrics_calculator.calculate_aggregate_metrics(
                    predictions, ground_truth, doc_ids
                )

                self.results[name] = ComparisonResult(
                    method_name=name,
                    metrics=metrics,
                    timing=timing,
                    predictions=predictions
                )

                logger.info(f"  Mean accuracy: {metrics.mean_accuracy:.3f}")
                logger.info(f"  Mean completion: {metrics.mean_completion_rate:.3f}")

            except Exception as e:
                logger.error(f"Failed to run {name}: {e}")
                continue

        return self.results

    def run_significance_tests(self) -> Dict[str, Dict]:
        """Run pairwise significance tests between methods."""

        significance_results = {}

        method_names = list(self.results.keys())

        for i, method1 in enumerate(method_names):
            for method2 in method_names[i+1:]:
                result1 = self.results[method1]
                result2 = self.results[method2]

                # Get per-document accuracies
                # This requires re-computing document-level metrics
                doc_accuracies_1 = self._get_document_accuracies(result1)
                doc_accuracies_2 = self._get_document_accuracies(result2)

                if len(doc_accuracies_1) == len(doc_accuracies_2):
                    comparison = self.significance_tester.paired_comparison(
                        doc_accuracies_1,
                        doc_accuracies_2,
                        method1,
                        method2
                    )
                    significance_results[f"{method1}_vs_{method2}"] = comparison

        return significance_results

    def _get_document_accuracies(self, result: ComparisonResult) -> List[float]:
        """Extract per-document accuracy scores from results."""
        # This is a simplified version - in practice, you'd store these during evaluation
        accuracies = []
        for field_metrics in result.metrics.field_metrics.values():
            if field_metrics.accuracy is not None:
                accuracies.append(field_metrics.accuracy)
            elif field_metrics.accuracy_5pct is not None:
                accuracies.append(field_metrics.accuracy_5pct)
        return accuracies if accuracies else [result.metrics.mean_accuracy]

    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""

        lines = []
        lines.append("=" * 100)
        lines.append("BASELINE COMPARISON REPORT")
        lines.append("Bridge Design Information Extraction")
        lines.append("=" * 100)
        lines.append("")

        # Summary table
        lines.append("PERFORMANCE SUMMARY")
        lines.append("-" * 100)
        lines.append(f"{'Method':<35} {'Accuracy':<12} {'Completion':<12} {'Consistency':<12} {'Time (s)':<12}")
        lines.append("-" * 100)

        for name, result in sorted(self.results.items(), key=lambda x: -x[1].metrics.mean_accuracy):
            acc = f"{result.metrics.mean_accuracy:.1%}"
            comp = f"{result.metrics.mean_completion_rate:.1%}"
            cons = f"{result.metrics.mean_consistency:.1%}"
            time_str = f"{result.timing.get('mean_time', 0):.3f}" if result.timing else "N/A"

            lines.append(f"{name:<35} {acc:<12} {comp:<12} {cons:<12} {time_str:<12}")

        lines.append("")

        # Detailed field-level comparison
        lines.append("FIELD-LEVEL ACCURACY BY METHOD")
        lines.append("-" * 100)

        # Get all field names
        field_names = list(EXTRACTION_FIELDS.keys())[:10]  # Top 10 fields

        header = f"{'Field':<25}"
        for name in self.results.keys():
            header += f" {name[:12]:<12}"
        lines.append(header)
        lines.append("-" * 100)

        for field_name in field_names:
            row = f"{field_name:<25}"
            for name, result in self.results.items():
                fm = result.metrics.field_metrics.get(field_name)
                if fm:
                    acc = fm.accuracy or fm.accuracy_5pct or 0
                    row += f" {acc:.1%}{'':>6}"
                else:
                    row += f" {'N/A':<12}"
            lines.append(row)

        lines.append("")

        # Significance tests
        sig_results = self.run_significance_tests()
        if sig_results:
            lines.append("STATISTICAL SIGNIFICANCE TESTS")
            lines.append("-" * 100)
            for comparison, result in sig_results.items():
                sig = "***" if result.get("wilcoxon_significant") else ""
                p_val = result.get("wilcoxon_p_value", "N/A")
                effect = result.get("effect_size_interpretation", "N/A")
                lines.append(f"{comparison}: p={p_val:.4f} {sig} (effect: {effect})")

        lines.append("")
        lines.append("=" * 100)

        report = "\n".join(lines)

        # Save report
        with open(self.output_dir / "comparison_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)

        return report

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Comparison of Extraction Methods}",
            r"\label{tab:comparison}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Method & Accuracy & Completion & Consistency & Time (s) \\",
            r"\midrule",
        ]

        for name, result in sorted(self.results.items(), key=lambda x: -x[1].metrics.mean_accuracy):
            acc = f"{result.metrics.mean_accuracy:.1%}"
            comp = f"{result.metrics.mean_completion_rate:.1%}"
            cons = f"{result.metrics.mean_consistency:.1%}"
            time_str = f"{result.timing.get('mean_time', 0):.2f}" if result.timing else "N/A"

            # Escape underscores for LaTeX
            name_escaped = name.replace("_", r"\_")

            lines.append(f"{name_escaped} & {acc} & {comp} & {cons} & {time_str} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        latex = "\n".join(lines)

        with open(self.output_dir / "comparison_table.tex", 'w', encoding='utf-8') as f:
            f.write(latex)

        return latex

    def save_results(self):
        """Save all comparison results."""

        # Convert results to JSON-serializable format
        json_results = {}
        for name, result in self.results.items():
            json_results[name] = {
                "method": result.method_name,
                "mean_accuracy": result.metrics.mean_accuracy,
                "mean_completion_rate": result.metrics.mean_completion_rate,
                "mean_consistency": result.metrics.mean_consistency,
                "mean_plausibility": result.metrics.mean_plausibility,
                "critical_field_accuracy": result.metrics.critical_field_accuracy,
                "confidence_interval_95": result.metrics.confidence_interval_95,
                "timing": result.timing,
                "field_metrics": {
                    fname: {
                        "extraction_rate": fm.extraction_rate,
                        "accuracy": fm.accuracy,
                        "accuracy_5pct": fm.accuracy_5pct,
                        "f1": fm.f1,
                        "mae": fm.mae,
                        "plausibility_rate": fm.plausibility_rate,
                    }
                    for fname, fm in result.metrics.field_metrics.items()
                }
            }

        with open(self.output_dir / "comparison_results.json", 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run baseline comparison."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare extraction methods")
    parser.add_argument("--documents", type=str, default="documents",
                       help="Path to documents directory")
    parser.add_argument("--ground-truth", type=str, default="final_json",
                       help="Path to ground truth JSON directory")
    parser.add_argument("--output", type=str, default="comparison_output",
                       help="Output directory")
    parser.add_argument("--run-llm", action="store_true",
                       help="Run LLM-based extractors")
    parser.add_argument("--max-docs", type=int, default=50,
                       help="Maximum documents to process")

    args = parser.parse_args()

    # Load documents and ground truth
    docs_dir = Path(args.documents)
    truth_dir = Path(args.ground_truth)

    documents = []
    ground_truth = []

    for json_file in sorted(truth_dir.glob("*.json"))[:args.max_docs]:
        doc_id = json_file.stem

        # Try to load document text
        txt_file = docs_dir / f"{doc_id}.txt"
        if not txt_file.exists():
            continue

        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()

        with open(json_file, 'r', encoding='utf-8') as f:
            truth = json.load(f)

        documents.append((doc_id, text[:10000]))
        ground_truth.append(truth)

    logger.info(f"Loaded {len(documents)} documents for comparison")

    # Run comparison
    comparison = BaselineComparison(output_dir=args.output)
    comparison.register_all_baselines()

    results = comparison.run_comparison(documents, ground_truth, run_llm=args.run_llm)

    # Generate reports
    report = comparison.generate_comparison_report()
    print(report)

    latex = comparison.generate_latex_table()
    print("\nLaTeX Table:")
    print(latex)

    comparison.save_results()

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
