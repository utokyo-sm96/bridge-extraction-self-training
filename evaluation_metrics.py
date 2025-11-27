"""
Comprehensive Evaluation Metrics for Bridge Design Information Extraction
==========================================================================

Provides extensive evaluation metrics for academic paper defense:
1. Field-level metrics (precision, recall, F1, accuracy)
2. Document-level metrics (exact match, partial match)
3. Engineering-specific metrics (plausibility, consistency)
4. Statistical significance testing
5. Error analysis and categorization

Author: Self-Training IE Framework
License: MIT
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, pearsonr
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# FIELD DEFINITIONS (imported from main framework)
# ============================================================================

EXTRACTION_FIELDS = {
    "girder_length": {"path": ["geometry", "girders", "length"], "type": "numeric", "unit": "mm", "critical": True},
    "num_girders": {"path": ["geometry", "girders", "num_girders"], "type": "integer", "unit": None, "critical": True},
    "spacing_x": {"path": ["geometry", "girders", "spacing_x"], "type": "numeric", "unit": "mm", "critical": False},
    "spacing_z": {"path": ["geometry", "girders", "spacing_z"], "type": "numeric", "unit": "mm", "critical": False},
    "top_flange_width": {"path": ["geometry", "girders", "top_flange_width"], "type": "numeric", "unit": "mm", "critical": False},
    "top_flange_thickness": {"path": ["geometry", "girders", "top_flange_thickness"], "type": "numeric", "unit": "mm", "critical": False},
    "bottom_flange_width": {"path": ["geometry", "girders", "bottom_flange_width"], "type": "numeric", "unit": "mm", "critical": False},
    "bottom_flange_thickness": {"path": ["geometry", "girders", "bottom_flange_thickness"], "type": "numeric", "unit": "mm", "critical": False},
    "web_height": {"path": ["geometry", "girders", "web_height"], "type": "numeric", "unit": "mm", "critical": True},
    "web_thickness": {"path": ["geometry", "girders", "web_thickness"], "type": "numeric", "unit": "mm", "critical": False},
    "x_offset": {"path": ["geometry", "girders", "x_offset"], "type": "numeric", "unit": "mm", "critical": False},
    "deck_length": {"path": ["geometry", "deck", "length"], "type": "numeric", "unit": "mm", "critical": False},
    "deck_width": {"path": ["geometry", "deck", "width"], "type": "numeric", "unit": "mm", "critical": False},
    "deck_thickness": {"path": ["geometry", "deck", "thickness"], "type": "numeric", "unit": "mm", "critical": True},
    "bridge_type": {"path": ["bridge_type"], "type": "categorical", "unit": None, "critical": True},
    "concrete_density": {"path": ["material_properties", "concrete", "density"], "type": "numeric", "unit": "N/m3", "critical": False},
    "steel_density": {"path": ["material_properties", "steel", "density"], "type": "numeric", "unit": "kg/m3", "critical": False},
    "steel_young_modulus": {"path": ["material_properties", "steel", "young_modulus"], "type": "numeric", "unit": "N/mm2", "critical": False},
    "live_load_type": {"path": ["live_load", "type"], "type": "categorical", "unit": None, "critical": False},
    "p1_bending": {"path": ["live_load", "p1_bending"], "type": "numeric", "unit": "kN", "critical": False},
    "p1_shear": {"path": ["live_load", "p1_shear"], "type": "numeric", "unit": "kN", "critical": False},
    "impact_coefficient": {"path": ["live_load", "impact_coefficient"], "type": "numeric", "unit": None, "critical": True},
    "use_crossbeams": {"path": ["crossbeams", "use_crossbeams"], "type": "boolean", "unit": None, "critical": False},
    "crossbeam_height": {"path": ["crossbeams", "height"], "type": "numeric", "unit": "mm", "critical": False},
    "crossbeam_thickness": {"path": ["crossbeams", "thickness"], "type": "numeric", "unit": "mm", "critical": False},
    "crossbeam_length": {"path": ["crossbeams", "length"], "type": "numeric", "unit": "mm", "critical": False},
    "crossbeam_spacing_z": {"path": ["crossbeams", "spacing_z"], "type": "numeric", "unit": "mm", "critical": False},
    "num_cross_girders": {"path": ["crossbeams", "num_cross_girders"], "type": "integer", "unit": None, "critical": False},
    "use_rebar": {"path": ["rebar", "use_rebar"], "type": "boolean", "unit": None, "critical": False},
    "rebar_spacing": {"path": ["rebar", "rebar_spacing"], "type": "numeric", "unit": "mm", "critical": False},
}

ENGINEERING_CONSTRAINTS = {
    "girder_length": (5000, 100000),
    "num_girders": (2, 10),
    "spacing_x": (1500, 15000),
    "spacing_z": (2000, 15000),
    "top_flange_width": (200, 1500),
    "top_flange_thickness": (10, 100),
    "bottom_flange_width": (200, 1500),
    "bottom_flange_thickness": (10, 100),
    "web_height": (500, 5000),
    "web_thickness": (8, 50),
    "deck_thickness": (150, 500),
    "deck_width": (3000, 30000),
    "crossbeam_height": (300, 3000),
    "crossbeam_thickness": (6, 30),
    "impact_coefficient": (0.1, 0.5),
}


# ============================================================================
# DATA CLASSES FOR RESULTS
# ============================================================================

@dataclass
class FieldMetrics:
    """Metrics for a single extraction field."""
    field_name: str
    field_type: str

    # Availability metrics
    extraction_rate: float = 0.0  # % of documents with non-null values

    # Numeric field metrics
    mae: Optional[float] = None  # Mean Absolute Error
    rmse: Optional[float] = None  # Root Mean Squared Error
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    r2: Optional[float] = None  # R-squared
    relative_error_mean: Optional[float] = None
    relative_error_std: Optional[float] = None

    # Categorical/Boolean metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    # Tolerance-based accuracy (for numeric)
    accuracy_1pct: Optional[float] = None  # Within 1% tolerance
    accuracy_5pct: Optional[float] = None  # Within 5% tolerance
    accuracy_10pct: Optional[float] = None  # Within 10% tolerance
    exact_match: Optional[float] = None  # Exact string match

    # Engineering plausibility
    plausibility_rate: Optional[float] = None  # % within engineering bounds

    # Sample size
    n_samples: int = 0


@dataclass
class DocumentMetrics:
    """Metrics for document-level extraction quality."""
    doc_id: str

    # Overall scores
    field_completion_rate: float = 0.0  # % of fields extracted
    critical_field_completion: float = 0.0  # % of critical fields extracted
    average_field_accuracy: float = 0.0

    # Per-field results
    field_results: Dict[str, Dict] = field(default_factory=dict)

    # Engineering consistency
    consistency_score: float = 0.0
    plausibility_score: float = 0.0


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all documents and fields."""

    # Field-level aggregates
    field_metrics: Dict[str, FieldMetrics] = field(default_factory=dict)

    # Document-level aggregates
    mean_completion_rate: float = 0.0
    mean_accuracy: float = 0.0
    mean_consistency: float = 0.0
    mean_plausibility: float = 0.0

    # Overall scores
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    weighted_f1: float = 0.0

    # Critical fields only
    critical_field_accuracy: float = 0.0

    # Statistical measures
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)

    # Sample info
    n_documents: int = 0
    n_fields: int = 0


# ============================================================================
# EVALUATION METRICS CALCULATOR
# ============================================================================

class EvaluationMetricsCalculator:
    """
    Comprehensive evaluation metrics calculator for information extraction.
    """

    def __init__(self):
        self.field_definitions = EXTRACTION_FIELDS
        self.constraints = ENGINEERING_CONSTRAINTS

    def _get_nested_value(self, data: Dict, path: List[str]) -> Any:
        """Get value from nested dictionary."""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _parse_numeric(self, value: Any) -> Optional[float]:
        """Parse value to numeric, handling various formats."""
        if value is None:
            return None
        try:
            # Remove units and whitespace
            value_str = str(value).strip()
            value_str = re.sub(r'[^\d.\-]', '', value_str)
            return float(value_str) if value_str else None
        except (ValueError, TypeError):
            return None

    def _parse_boolean(self, value: Any) -> Optional[bool]:
        """Parse value to boolean."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        value_str = str(value).lower().strip()
        if value_str in ['true', '1', 'yes', 'on']:
            return True
        elif value_str in ['false', '0', 'no', 'off']:
            return False
        return None

    def calculate_field_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        field_name: str
    ) -> FieldMetrics:
        """
        Calculate comprehensive metrics for a single field.

        Args:
            predictions: List of predicted JSON documents
            ground_truth: List of ground truth JSON documents
            field_name: Name of the field to evaluate

        Returns:
            FieldMetrics object with all computed metrics
        """
        field_info = self.field_definitions.get(field_name)
        if not field_info:
            return FieldMetrics(field_name=field_name, field_type="unknown")

        field_type = field_info["type"]
        path = field_info["path"]

        metrics = FieldMetrics(field_name=field_name, field_type=field_type)

        # Collect paired values
        pred_values = []
        true_values = []

        for pred, truth in zip(predictions, ground_truth):
            pred_val = self._get_nested_value(pred, path)
            true_val = self._get_nested_value(truth, path)

            if true_val is not None:  # Only count where ground truth exists
                pred_values.append(pred_val)
                true_values.append(true_val)

        if not true_values:
            return metrics

        metrics.n_samples = len(true_values)

        # Extraction rate
        non_null_preds = sum(1 for v in pred_values if v is not None)
        metrics.extraction_rate = non_null_preds / len(pred_values)

        # Type-specific metrics
        if field_type in ["numeric", "integer"]:
            metrics = self._compute_numeric_metrics(metrics, pred_values, true_values, field_name)
        elif field_type == "categorical":
            metrics = self._compute_categorical_metrics(metrics, pred_values, true_values)
        elif field_type == "boolean":
            metrics = self._compute_boolean_metrics(metrics, pred_values, true_values)

        return metrics

    def _compute_numeric_metrics(
        self,
        metrics: FieldMetrics,
        pred_values: List,
        true_values: List,
        field_name: str
    ) -> FieldMetrics:
        """Compute metrics for numeric fields."""

        # Parse to numeric
        pred_numeric = [self._parse_numeric(v) for v in pred_values]
        true_numeric = [self._parse_numeric(v) for v in true_values]

        # Filter to valid pairs
        valid_pairs = [
            (p, t) for p, t in zip(pred_numeric, true_numeric)
            if p is not None and t is not None
        ]

        if not valid_pairs:
            return metrics

        preds = np.array([p for p, t in valid_pairs])
        trues = np.array([t for p, t in valid_pairs])

        # Basic regression metrics
        metrics.mae = float(mean_absolute_error(trues, preds))
        metrics.rmse = float(np.sqrt(mean_squared_error(trues, preds)))

        # R-squared (handle edge cases)
        if np.var(trues) > 0:
            metrics.r2 = float(r2_score(trues, preds))

        # Relative errors
        relative_errors = []
        for p, t in valid_pairs:
            if t != 0:
                rel_err = abs(p - t) / abs(t)
                relative_errors.append(rel_err)

        if relative_errors:
            metrics.relative_error_mean = float(np.mean(relative_errors))
            metrics.relative_error_std = float(np.std(relative_errors))
            metrics.mape = float(np.mean(relative_errors) * 100)

        # Tolerance-based accuracy
        metrics.accuracy_1pct = float(np.mean([
            abs(p - t) / max(abs(t), 1) <= 0.01 for p, t in valid_pairs
        ]))
        metrics.accuracy_5pct = float(np.mean([
            abs(p - t) / max(abs(t), 1) <= 0.05 for p, t in valid_pairs
        ]))
        metrics.accuracy_10pct = float(np.mean([
            abs(p - t) / max(abs(t), 1) <= 0.10 for p, t in valid_pairs
        ]))

        # Exact match (string comparison for integers)
        exact_matches = sum(
            1 for pv, tv in zip(pred_values, true_values)
            if pv is not None and tv is not None and str(pv).strip() == str(tv).strip()
        )
        metrics.exact_match = exact_matches / len(valid_pairs) if valid_pairs else 0.0

        # Engineering plausibility
        if field_name in self.constraints:
            min_val, max_val = self.constraints[field_name]
            plausible = sum(1 for p in preds if min_val <= p <= max_val)
            metrics.plausibility_rate = plausible / len(preds)

        return metrics

    def _compute_categorical_metrics(
        self,
        metrics: FieldMetrics,
        pred_values: List,
        true_values: List
    ) -> FieldMetrics:
        """Compute metrics for categorical fields."""

        # Normalize strings
        pred_normalized = [
            str(v).lower().strip() if v is not None else None
            for v in pred_values
        ]
        true_normalized = [
            str(v).lower().strip() if v is not None else None
            for v in true_values
        ]

        # Filter valid pairs
        valid_pairs = [
            (p, t) for p, t in zip(pred_normalized, true_normalized)
            if p is not None and t is not None
        ]

        if not valid_pairs:
            return metrics

        preds = [p for p, t in valid_pairs]
        trues = [t for p, t in valid_pairs]

        # Get unique labels
        labels = list(set(preds + trues))

        # Compute classification metrics
        metrics.accuracy = float(accuracy_score(trues, preds))

        if len(labels) > 1:
            metrics.precision = float(precision_score(trues, preds, average='weighted', zero_division=0))
            metrics.recall = float(recall_score(trues, preds, average='weighted', zero_division=0))
            metrics.f1 = float(f1_score(trues, preds, average='weighted', zero_division=0))
        else:
            metrics.precision = metrics.accuracy
            metrics.recall = metrics.accuracy
            metrics.f1 = metrics.accuracy

        # Exact match
        metrics.exact_match = metrics.accuracy

        return metrics

    def _compute_boolean_metrics(
        self,
        metrics: FieldMetrics,
        pred_values: List,
        true_values: List
    ) -> FieldMetrics:
        """Compute metrics for boolean fields."""

        pred_bool = [self._parse_boolean(v) for v in pred_values]
        true_bool = [self._parse_boolean(v) for v in true_values]

        # Filter valid pairs
        valid_pairs = [
            (p, t) for p, t in zip(pred_bool, true_bool)
            if p is not None and t is not None
        ]

        if not valid_pairs:
            return metrics

        preds = [int(p) for p, t in valid_pairs]
        trues = [int(t) for p, t in valid_pairs]

        metrics.accuracy = float(accuracy_score(trues, preds))
        metrics.precision = float(precision_score(trues, preds, zero_division=0))
        metrics.recall = float(recall_score(trues, preds, zero_division=0))
        metrics.f1 = float(f1_score(trues, preds, zero_division=0))
        metrics.exact_match = metrics.accuracy

        return metrics

    def calculate_document_metrics(
        self,
        prediction: Dict,
        ground_truth: Dict,
        doc_id: str
    ) -> DocumentMetrics:
        """Calculate metrics for a single document."""

        metrics = DocumentMetrics(doc_id=doc_id)

        total_fields = 0
        extracted_fields = 0
        critical_total = 0
        critical_extracted = 0
        accuracies = []
        plausibility_scores = []

        for field_name, field_info in self.field_definitions.items():
            path = field_info["path"]
            is_critical = field_info.get("critical", False)

            pred_val = self._get_nested_value(prediction, path)
            true_val = self._get_nested_value(ground_truth, path)

            if true_val is not None:
                total_fields += 1
                if is_critical:
                    critical_total += 1

                if pred_val is not None:
                    extracted_fields += 1
                    if is_critical:
                        critical_extracted += 1

                    # Calculate field accuracy
                    field_type = field_info["type"]
                    if field_type in ["numeric", "integer"]:
                        pred_num = self._parse_numeric(pred_val)
                        true_num = self._parse_numeric(true_val)
                        if pred_num is not None and true_num is not None and true_num != 0:
                            rel_error = abs(pred_num - true_num) / abs(true_num)
                            accuracy = max(0, 1 - rel_error)
                            accuracies.append(accuracy)

                            # Plausibility check
                            if field_name in self.constraints:
                                min_v, max_v = self.constraints[field_name]
                                if min_v <= pred_num <= max_v:
                                    plausibility_scores.append(1.0)
                                else:
                                    plausibility_scores.append(0.0)
                    else:
                        # Categorical/Boolean - exact match
                        if str(pred_val).lower().strip() == str(true_val).lower().strip():
                            accuracies.append(1.0)
                        else:
                            accuracies.append(0.0)

                    metrics.field_results[field_name] = {
                        "predicted": pred_val,
                        "ground_truth": true_val,
                        "match": str(pred_val).lower().strip() == str(true_val).lower().strip()
                    }

        # Compute aggregate metrics
        metrics.field_completion_rate = extracted_fields / total_fields if total_fields > 0 else 0
        metrics.critical_field_completion = critical_extracted / critical_total if critical_total > 0 else 0
        metrics.average_field_accuracy = np.mean(accuracies) if accuracies else 0
        metrics.plausibility_score = np.mean(plausibility_scores) if plausibility_scores else 1.0

        # Consistency checks
        metrics.consistency_score = self._check_document_consistency(prediction)

        return metrics

    def _check_document_consistency(self, prediction: Dict) -> float:
        """Check internal consistency of extracted values."""
        consistency_checks = []

        # Check 1: deck_length ≈ girder_length
        deck_length = self._parse_numeric(
            self._get_nested_value(prediction, ["geometry", "deck", "length"])
        )
        girder_length = self._parse_numeric(
            self._get_nested_value(prediction, ["geometry", "girders", "length"])
        )
        if deck_length and girder_length:
            if abs(deck_length - girder_length) < 100:
                consistency_checks.append(1.0)
            else:
                consistency_checks.append(0.5)

        # Check 2: crossbeam_length ≈ spacing_x
        cb_length = self._parse_numeric(
            self._get_nested_value(prediction, ["crossbeams", "length"])
        )
        spacing_x = self._parse_numeric(
            self._get_nested_value(prediction, ["geometry", "girders", "spacing_x"])
        )
        if cb_length and spacing_x:
            if abs(cb_length - spacing_x) < 500:
                consistency_checks.append(1.0)
            else:
                consistency_checks.append(0.5)

        # Check 3: web_height > flange_thickness * 10
        web_height = self._parse_numeric(
            self._get_nested_value(prediction, ["geometry", "girders", "web_height"])
        )
        flange_t = self._parse_numeric(
            self._get_nested_value(prediction, ["geometry", "girders", "top_flange_thickness"])
        )
        if web_height and flange_t:
            if web_height > flange_t * 10:
                consistency_checks.append(1.0)
            else:
                consistency_checks.append(0.3)

        # Check 4: top_flange ≈ bottom_flange (typically equal)
        top_flange_w = self._parse_numeric(
            self._get_nested_value(prediction, ["geometry", "girders", "top_flange_width"])
        )
        bot_flange_w = self._parse_numeric(
            self._get_nested_value(prediction, ["geometry", "girders", "bottom_flange_width"])
        )
        if top_flange_w and bot_flange_w:
            if abs(top_flange_w - bot_flange_w) < 100:
                consistency_checks.append(1.0)
            else:
                consistency_checks.append(0.7)  # Different but acceptable

        return np.mean(consistency_checks) if consistency_checks else 1.0

    def calculate_aggregate_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        doc_ids: Optional[List[str]] = None
    ) -> AggregateMetrics:
        """Calculate aggregate metrics across all documents and fields."""

        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(predictions))]

        aggregate = AggregateMetrics()
        aggregate.n_documents = len(predictions)
        aggregate.n_fields = len(self.field_definitions)

        # Calculate field-level metrics
        for field_name in self.field_definitions:
            field_metrics = self.calculate_field_metrics(predictions, ground_truth, field_name)
            aggregate.field_metrics[field_name] = field_metrics

        # Calculate document-level metrics
        doc_completions = []
        doc_accuracies = []
        doc_consistencies = []
        doc_plausibilities = []

        for pred, truth, doc_id in zip(predictions, ground_truth, doc_ids):
            doc_metrics = self.calculate_document_metrics(pred, truth, doc_id)
            doc_completions.append(doc_metrics.field_completion_rate)
            doc_accuracies.append(doc_metrics.average_field_accuracy)
            doc_consistencies.append(doc_metrics.consistency_score)
            doc_plausibilities.append(doc_metrics.plausibility_score)

        aggregate.mean_completion_rate = float(np.mean(doc_completions))
        aggregate.mean_accuracy = float(np.mean(doc_accuracies))
        aggregate.mean_consistency = float(np.mean(doc_consistencies))
        aggregate.mean_plausibility = float(np.mean(doc_plausibilities))

        # Compute macro/micro/weighted F1
        all_f1s = [
            m.f1 for m in aggregate.field_metrics.values()
            if m.f1 is not None
        ]
        aggregate.macro_f1 = float(np.mean(all_f1s)) if all_f1s else 0.0

        # Critical field accuracy
        critical_accuracies = []
        for field_name, field_info in self.field_definitions.items():
            if field_info.get("critical", False):
                fm = aggregate.field_metrics.get(field_name)
                if fm:
                    if fm.accuracy is not None:
                        critical_accuracies.append(fm.accuracy)
                    elif fm.accuracy_5pct is not None:
                        critical_accuracies.append(fm.accuracy_5pct)

        aggregate.critical_field_accuracy = float(np.mean(critical_accuracies)) if critical_accuracies else 0.0

        # 95% confidence interval for mean accuracy
        if len(doc_accuracies) > 1:
            ci = stats.t.interval(
                0.95,
                len(doc_accuracies) - 1,
                loc=np.mean(doc_accuracies),
                scale=stats.sem(doc_accuracies)
            )
            aggregate.confidence_interval_95 = (float(ci[0]), float(ci[1]))

        return aggregate

    def generate_report(
        self,
        aggregate_metrics: AggregateMetrics,
        output_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive evaluation report."""

        lines = []
        lines.append("=" * 80)
        lines.append("COMPREHENSIVE EVALUATION REPORT")
        lines.append("Bridge Design Information Extraction")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Documents evaluated: {aggregate_metrics.n_documents}")
        lines.append(f"Fields evaluated: {aggregate_metrics.n_fields}")
        lines.append(f"Mean field completion rate: {aggregate_metrics.mean_completion_rate:.1%}")
        lines.append(f"Mean extraction accuracy: {aggregate_metrics.mean_accuracy:.1%}")
        lines.append(f"Critical field accuracy: {aggregate_metrics.critical_field_accuracy:.1%}")
        lines.append(f"95% CI for accuracy: [{aggregate_metrics.confidence_interval_95[0]:.1%}, {aggregate_metrics.confidence_interval_95[1]:.1%}]")
        lines.append("")

        # Engineering quality
        lines.append("ENGINEERING QUALITY METRICS")
        lines.append("-" * 40)
        lines.append(f"Mean consistency score: {aggregate_metrics.mean_consistency:.1%}")
        lines.append(f"Mean plausibility score: {aggregate_metrics.mean_plausibility:.1%}")
        lines.append("")

        # Field-level details
        lines.append("FIELD-LEVEL METRICS")
        lines.append("-" * 40)
        lines.append(f"{'Field':<30} {'Type':<12} {'Extraction':<12} {'Accuracy':<12} {'Plausibility':<12}")
        lines.append("-" * 80)

        for field_name, fm in sorted(aggregate_metrics.field_metrics.items()):
            acc_str = ""
            if fm.accuracy is not None:
                acc_str = f"{fm.accuracy:.1%}"
            elif fm.accuracy_5pct is not None:
                acc_str = f"{fm.accuracy_5pct:.1%} (5%)"

            plaus_str = f"{fm.plausibility_rate:.1%}" if fm.plausibility_rate is not None else "N/A"

            lines.append(
                f"{field_name:<30} {fm.field_type:<12} {fm.extraction_rate:.1%}{'':>6} "
                f"{acc_str:<12} {plaus_str:<12}"
            )

        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report


# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

class StatisticalSignificanceTester:
    """
    Performs statistical significance tests for comparing extraction methods.
    Essential for academic paper defense.
    """

    def __init__(self):
        self.alpha = 0.05  # Significance level

    def paired_comparison(
        self,
        method1_scores: List[float],
        method2_scores: List[float],
        method1_name: str = "Method 1",
        method2_name: str = "Method 2"
    ) -> Dict[str, Any]:
        """
        Perform paired comparison between two methods.

        Uses:
        - Wilcoxon signed-rank test (non-parametric)
        - Paired t-test (parametric)
        - Effect size (Cohen's d)
        """

        scores1 = np.array(method1_scores)
        scores2 = np.array(method2_scores)

        results = {
            "method1_name": method1_name,
            "method2_name": method2_name,
            "n_samples": len(scores1),
            "method1_mean": float(np.mean(scores1)),
            "method1_std": float(np.std(scores1)),
            "method2_mean": float(np.mean(scores2)),
            "method2_std": float(np.std(scores2)),
            "mean_difference": float(np.mean(scores2) - np.mean(scores1)),
        }

        # Wilcoxon signed-rank test
        try:
            stat, p_value = wilcoxon(scores1, scores2)
            results["wilcoxon_statistic"] = float(stat)
            results["wilcoxon_p_value"] = float(p_value)
            results["wilcoxon_significant"] = p_value < self.alpha
        except Exception as e:
            results["wilcoxon_error"] = str(e)

        # Paired t-test
        try:
            stat, p_value = stats.ttest_rel(scores1, scores2)
            results["ttest_statistic"] = float(stat)
            results["ttest_p_value"] = float(p_value)
            results["ttest_significant"] = p_value < self.alpha
        except Exception as e:
            results["ttest_error"] = str(e)

        # Effect size (Cohen's d)
        diff = scores2 - scores1
        pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
        if pooled_std > 0:
            cohens_d = np.mean(diff) / pooled_std
            results["cohens_d"] = float(cohens_d)

            # Interpret effect size
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                interpretation = "negligible"
            elif abs_d < 0.5:
                interpretation = "small"
            elif abs_d < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"
            results["effect_size_interpretation"] = interpretation

        return results

    def bootstrap_confidence_interval(
        self,
        scores: List[float],
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for mean.
        """
        scores = np.array(scores)
        n = len(scores)

        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(resample))

        bootstrap_means = np.array(bootstrap_means)

        # Compute percentiles
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap
        }

    def correlation_analysis(
        self,
        confidence_scores: List[float],
        accuracy_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Analyze correlation between confidence and accuracy.
        Validates that confidence scores are meaningful.
        """

        conf = np.array(confidence_scores)
        acc = np.array(accuracy_scores)

        results = {}

        # Pearson correlation (linear)
        r, p = pearsonr(conf, acc)
        results["pearson_r"] = float(r)
        results["pearson_p"] = float(p)
        results["pearson_significant"] = p < self.alpha

        # Spearman correlation (monotonic)
        rho, p = spearmanr(conf, acc)
        results["spearman_rho"] = float(rho)
        results["spearman_p"] = float(p)
        results["spearman_significant"] = p < self.alpha

        # Interpretation
        if abs(r) > 0.7:
            strength = "strong"
        elif abs(r) > 0.4:
            strength = "moderate"
        elif abs(r) > 0.2:
            strength = "weak"
        else:
            strength = "negligible"

        results["correlation_strength"] = strength
        results["correlation_direction"] = "positive" if r > 0 else "negative"

        return results


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

class ErrorAnalyzer:
    """
    Performs detailed error analysis for understanding extraction failures.
    """

    def __init__(self):
        self.field_definitions = EXTRACTION_FIELDS

    def _get_nested_value(self, data: Dict, path: List[str]) -> Any:
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def categorize_errors(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
        doc_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Categorize extraction errors into types.

        Error categories:
        1. Missing extraction (null prediction, non-null ground truth)
        2. Spurious extraction (non-null prediction, null ground truth)
        3. Wrong value (both non-null, values differ)
        4. Type error (wrong data type)
        5. Unit error (value off by factor of 1000, 10, etc.)
        """

        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(predictions))]

        error_categories = {
            "missing_extraction": [],
            "spurious_extraction": [],
            "wrong_value": [],
            "magnitude_error": [],  # Off by factor of 10, 100, 1000
            "sign_error": [],
            "partial_match": [],
        }

        field_error_counts = defaultdict(lambda: defaultdict(int))

        for doc_idx, (pred, truth, doc_id) in enumerate(zip(predictions, ground_truth, doc_ids)):
            for field_name, field_info in self.field_definitions.items():
                path = field_info["path"]
                pred_val = self._get_nested_value(pred, path)
                true_val = self._get_nested_value(truth, path)

                # Skip if ground truth is null
                if true_val is None:
                    if pred_val is not None:
                        error_categories["spurious_extraction"].append({
                            "doc_id": doc_id,
                            "field": field_name,
                            "predicted": pred_val
                        })
                        field_error_counts[field_name]["spurious"] += 1
                    continue

                # Missing extraction
                if pred_val is None:
                    error_categories["missing_extraction"].append({
                        "doc_id": doc_id,
                        "field": field_name,
                        "ground_truth": true_val
                    })
                    field_error_counts[field_name]["missing"] += 1
                    continue

                # Check if values match
                if str(pred_val).strip() == str(true_val).strip():
                    continue  # Correct

                # Analyze the error type
                if field_info["type"] in ["numeric", "integer"]:
                    try:
                        pred_num = float(str(pred_val).replace(',', ''))
                        true_num = float(str(true_val).replace(',', ''))

                        # Check for magnitude errors
                        if true_num != 0:
                            ratio = pred_num / true_num
                            if 0.99 <= ratio <= 1.01:
                                continue  # Within 1% - correct
                            elif ratio in [10, 100, 1000, 0.1, 0.01, 0.001]:
                                error_categories["magnitude_error"].append({
                                    "doc_id": doc_id,
                                    "field": field_name,
                                    "predicted": pred_val,
                                    "ground_truth": true_val,
                                    "ratio": ratio
                                })
                                field_error_counts[field_name]["magnitude"] += 1
                                continue

                        # Sign error
                        if pred_num * true_num < 0:
                            error_categories["sign_error"].append({
                                "doc_id": doc_id,
                                "field": field_name,
                                "predicted": pred_val,
                                "ground_truth": true_val
                            })
                            field_error_counts[field_name]["sign"] += 1
                            continue

                    except (ValueError, TypeError):
                        pass

                # Check for partial match (substring)
                pred_str = str(pred_val).lower()
                true_str = str(true_val).lower()
                if pred_str in true_str or true_str in pred_str:
                    error_categories["partial_match"].append({
                        "doc_id": doc_id,
                        "field": field_name,
                        "predicted": pred_val,
                        "ground_truth": true_val
                    })
                    field_error_counts[field_name]["partial"] += 1
                    continue

                # General wrong value
                error_categories["wrong_value"].append({
                    "doc_id": doc_id,
                    "field": field_name,
                    "predicted": pred_val,
                    "ground_truth": true_val
                })
                field_error_counts[field_name]["wrong"] += 1

        # Compute summary statistics
        summary = {
            "total_errors": sum(len(v) for v in error_categories.values()),
            "error_counts_by_category": {k: len(v) for k, v in error_categories.items()},
            "error_counts_by_field": dict(field_error_counts),
            "most_error_prone_fields": sorted(
                field_error_counts.keys(),
                key=lambda f: sum(field_error_counts[f].values()),
                reverse=True
            )[:10]
        }

        return {
            "errors": error_categories,
            "summary": summary
        }

    def generate_error_report(
        self,
        error_analysis: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate a detailed error analysis report."""

        lines = []
        lines.append("=" * 80)
        lines.append("ERROR ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        summary = error_analysis["summary"]

        lines.append("ERROR SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total errors: {summary['total_errors']}")
        lines.append("")
        lines.append("Errors by category:")
        for category, count in summary["error_counts_by_category"].items():
            lines.append(f"  {category}: {count}")
        lines.append("")

        lines.append("Most error-prone fields:")
        for field in summary["most_error_prone_fields"]:
            counts = summary["error_counts_by_field"][field]
            total = sum(counts.values())
            lines.append(f"  {field}: {total} errors")
            for err_type, cnt in counts.items():
                lines.append(f"    - {err_type}: {cnt}")
        lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

        return report


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def run_comprehensive_evaluation(
    predictions_path: str,
    ground_truth_path: str,
    output_dir: str = "evaluation_output"
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on extraction results.

    Args:
        predictions_path: Path to directory with predicted JSON files
        ground_truth_path: Path to directory with ground truth JSON files
        output_dir: Output directory for reports

    Returns:
        Dictionary with all evaluation results
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    pred_dir = Path(predictions_path)
    truth_dir = Path(ground_truth_path)

    predictions = []
    ground_truth = []
    doc_ids = []

    for pred_file in sorted(pred_dir.glob("*.json")):
        doc_id = pred_file.stem
        truth_file = truth_dir / f"{doc_id}.json"

        if truth_file.exists():
            with open(pred_file, 'r', encoding='utf-8') as f:
                predictions.append(json.load(f))
            with open(truth_file, 'r', encoding='utf-8') as f:
                ground_truth.append(json.load(f))
            doc_ids.append(doc_id)

    logger.info(f"Loaded {len(predictions)} document pairs for evaluation")

    # Calculate metrics
    calculator = EvaluationMetricsCalculator()
    aggregate_metrics = calculator.calculate_aggregate_metrics(predictions, ground_truth, doc_ids)

    # Generate report
    report = calculator.generate_report(
        aggregate_metrics,
        output_path=str(output_dir / "evaluation_report.txt")
    )
    print(report)

    # Error analysis
    analyzer = ErrorAnalyzer()
    error_analysis = analyzer.categorize_errors(predictions, ground_truth, doc_ids)
    error_report = analyzer.generate_error_report(
        error_analysis,
        output_path=str(output_dir / "error_analysis.txt")
    )

    # Save all results
    results = {
        "aggregate_metrics": {
            "mean_accuracy": aggregate_metrics.mean_accuracy,
            "mean_completion_rate": aggregate_metrics.mean_completion_rate,
            "mean_consistency": aggregate_metrics.mean_consistency,
            "mean_plausibility": aggregate_metrics.mean_plausibility,
            "critical_field_accuracy": aggregate_metrics.critical_field_accuracy,
            "confidence_interval_95": aggregate_metrics.confidence_interval_95,
            "n_documents": aggregate_metrics.n_documents,
        },
        "field_metrics": {
            name: {
                "extraction_rate": fm.extraction_rate,
                "accuracy": fm.accuracy,
                "accuracy_5pct": fm.accuracy_5pct,
                "f1": fm.f1,
                "mae": fm.mae,
                "rmse": fm.rmse,
                "plausibility_rate": fm.plausibility_rate,
            }
            for name, fm in aggregate_metrics.field_metrics.items()
        },
        "error_summary": error_analysis["summary"],
    }

    with open(output_dir / "evaluation_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive evaluation of extraction results")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions directory")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth directory")
    parser.add_argument("--output", type=str, default="evaluation_output", help="Output directory")

    args = parser.parse_args()

    run_comprehensive_evaluation(args.predictions, args.ground_truth, args.output)
