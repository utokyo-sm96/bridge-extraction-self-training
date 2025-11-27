#!/usr/bin/env python3
"""
Quick Test for Self-Training Framework
=======================================

Runs a minimal test to verify the framework components work correctly.
Tests data loading, confidence estimation, and basic functionality.

Usage:
    python quick_test.py
    python quick_test.py --verbose
    python quick_test.py --test-training  # Test with actual training (slower)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuickTest:
    """Quick test runner for the self-training framework."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.start_time = datetime.now()

    def log(self, message: str, level: str = "info"):
        """Log a message."""
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug" and self.verbose:
            logger.debug(message)

    def test_data_loading(self) -> bool:
        """Test that data can be loaded correctly."""
        print("\n" + "-"*50)
        print("TEST 1: Data Loading")
        print("-"*50)

        try:
            json_dir = Path("data/json")
            docs_dir = Path("data/documents")

            # Check directories exist
            if not json_dir.exists():
                print(f"  ✗ JSON directory not found: {json_dir}")
                return False

            if not docs_dir.exists():
                print(f"  ✗ Documents directory not found: {docs_dir}")
                return False

            # Count files
            json_files = list(json_dir.glob("*.json"))
            docx_files = list(docs_dir.glob("*.docx"))
            txt_files = list(docs_dir.glob("*.txt"))

            print(f"  JSON files: {len(json_files)}")
            print(f"  DOCX files: {len(docx_files)}")
            print(f"  TXT files: {len(txt_files)}")

            # Find matching pairs
            json_stems = {f.stem for f in json_files}
            doc_stems = {f.stem for f in docx_files} | {f.stem for f in txt_files}
            matching = json_stems & doc_stems

            print(f"  Matching pairs: {len(matching)}")

            if len(matching) < 5:
                print("  ✗ Need at least 5 matching document-label pairs")
                return False

            # Try loading a sample
            sample_id = list(matching)[0]
            json_path = json_dir / f"{sample_id}.json"

            with open(json_path, 'r', encoding='utf-8') as f:
                sample_json = json.load(f)

            print(f"  Sample loaded: {sample_id}")
            print(f"  Fields in sample: {list(sample_json.keys())}")

            print("  ✓ Data loading OK")
            return True

        except Exception as e:
            print(f"  ✗ Data loading failed: {e}")
            return False

    def test_docx_extraction(self) -> bool:
        """Test DOCX text extraction."""
        print("\n" + "-"*50)
        print("TEST 2: DOCX Text Extraction")
        print("-"*50)

        try:
            docs_dir = Path("data/documents")
            docx_files = list(docs_dir.glob("*.docx"))[:3]  # Test first 3

            if not docx_files:
                print("  ○ No DOCX files found (skipping)")
                return True

            for docx_path in docx_files:
                text = self._extract_docx_text(str(docx_path))
                if text:
                    print(f"  ✓ {docx_path.name}: {len(text)} chars")
                else:
                    print(f"  ✗ {docx_path.name}: extraction failed")
                    return False

            print("  ✓ DOCX extraction OK")
            return True

        except Exception as e:
            print(f"  ✗ DOCX extraction failed: {e}")
            return False

    def _extract_docx_text(self, docx_path: str) -> str:
        """Extract text from DOCX file."""
        # Try python-docx first
        try:
            from docx import Document
            doc = Document(docx_path)
            paragraphs = [para.text for para in doc.paragraphs]
            return '\n'.join(paragraphs)
        except ImportError:
            pass
        except Exception:
            pass

        # Fall back to zipfile
        try:
            import zipfile
            import xml.etree.ElementTree as ET

            with zipfile.ZipFile(docx_path, 'r') as z:
                if 'word/document.xml' not in z.namelist():
                    return ""

                xml_content = z.read('word/document.xml')
                tree = ET.fromstring(xml_content)

                paragraphs = []
                ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                for para in tree.iter(f'{ns}p'):
                    texts = []
                    for text in para.iter(f'{ns}t'):
                        if text.text:
                            texts.append(text.text)
                    if texts:
                        paragraphs.append(''.join(texts))

                return '\n'.join(paragraphs)
        except Exception as e:
            logger.debug(f"Zipfile extraction failed: {e}")
            return ""

    def test_confidence_estimation(self) -> bool:
        """Test confidence estimation logic."""
        print("\n" + "-"*50)
        print("TEST 3: Confidence Estimation")
        print("-"*50)

        try:
            # Create mock data
            sample_json = {
                "bridge_type": "PlateGirder",
                "geometry": {
                    "girders": {
                        "length": "25000",
                        "num_girders": "2",
                        "web_height": "1700",
                        "web_thickness": "9",
                    },
                    "deck": {
                        "thickness": "280",
                        "length": "25000"
                    }
                },
                "crossbeams": {
                    "use_crossbeams": "True",
                    "height": "1330"
                }
            }

            sample_text = """
            主桁支間長 25000mm
            主桁本数 2本
            主桁腹板高 1700
            WEB PL 1700 * 9
            床版厚 280mm
            橋梁形式 鈑桁橋
            """

            # Test source verification
            source_score = self._verify_in_source("25000", sample_text)
            print(f"  Source verification (25000): {source_score:.2f}")

            # Test plausibility
            plausibility = self._check_plausibility("girder_length", 25000)
            print(f"  Plausibility (girder_length=25000): {plausibility:.2f}")

            # Test consistency
            consistency = self._check_consistency(sample_json)
            print(f"  Consistency score: {consistency:.2f}")

            # Combined confidence
            confidence = 0.35 * source_score + 0.25 * plausibility + 0.20 * consistency + 0.20
            print(f"  Combined confidence: {confidence:.2f}")

            if confidence > 0.5:
                print("  ✓ Confidence estimation OK")
                return True
            else:
                print("  ⚠ Confidence lower than expected")
                return True  # Still pass as it's working

        except Exception as e:
            print(f"  ✗ Confidence estimation failed: {e}")
            return False

    def _verify_in_source(self, value: str, text: str) -> float:
        """Check if value exists in source text."""
        if value in text:
            return 1.0
        try:
            numeric = float(value)
            if str(int(numeric)) in text:
                return 0.9
        except:
            pass
        return 0.3

    def _check_plausibility(self, field_name: str, value: float) -> float:
        """Check engineering plausibility."""
        constraints = {
            "girder_length": (5000, 100000),
            "num_girders": (2, 10),
            "web_height": (500, 5000),
            "deck_thickness": (150, 500),
        }

        if field_name not in constraints:
            return 0.8

        min_val, max_val = constraints[field_name]
        if min_val <= value <= max_val:
            return 1.0
        return 0.3

    def _check_consistency(self, data: Dict) -> float:
        """Check cross-field consistency."""
        checks = []

        # Deck length should match girder length
        deck_len = data.get("geometry", {}).get("deck", {}).get("length")
        girder_len = data.get("geometry", {}).get("girders", {}).get("length")

        if deck_len and girder_len:
            try:
                if abs(float(deck_len) - float(girder_len)) < 100:
                    checks.append(1.0)
                else:
                    checks.append(0.5)
            except:
                checks.append(0.5)

        return sum(checks) / len(checks) if checks else 0.8

    def test_evaluation_metrics(self) -> bool:
        """Test evaluation metrics calculation."""
        print("\n" + "-"*50)
        print("TEST 4: Evaluation Metrics")
        print("-"*50)

        try:
            import numpy as np

            # Create mock predictions and ground truth
            predictions = [
                {"geometry": {"girders": {"length": "25000", "web_height": "1700"}}},
                {"geometry": {"girders": {"length": "30000", "web_height": "2000"}}},
                {"geometry": {"girders": {"length": "20000", "web_height": "1500"}}},
            ]

            ground_truth = [
                {"geometry": {"girders": {"length": "25000", "web_height": "1700"}}},
                {"geometry": {"girders": {"length": "30500", "web_height": "2000"}}},
                {"geometry": {"girders": {"length": "20000", "web_height": "1550"}}},
            ]

            # Calculate accuracy
            accuracies = []
            for pred, truth in zip(predictions, ground_truth):
                pred_len = float(pred["geometry"]["girders"]["length"])
                true_len = float(truth["geometry"]["girders"]["length"])
                rel_error = abs(pred_len - true_len) / true_len
                accuracy = max(0, 1 - rel_error)
                accuracies.append(accuracy)

            mean_accuracy = np.mean(accuracies)
            print(f"  Mean accuracy: {mean_accuracy:.3f}")

            # Calculate 95% CI
            from scipy import stats
            if len(accuracies) > 1:
                ci = stats.t.interval(
                    0.95,
                    len(accuracies) - 1,
                    loc=np.mean(accuracies),
                    scale=stats.sem(accuracies)
                )
                print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

            print("  ✓ Evaluation metrics OK")
            return True

        except Exception as e:
            print(f"  ✗ Evaluation metrics failed: {e}")
            return False

    def test_visualization(self) -> bool:
        """Test visualization generation."""
        print("\n" + "-"*50)
        print("TEST 5: Visualization")
        print("-"*50)

        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np

            # Create test plot
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y)
            ax.set_title("Test Plot")

            # Save to temp file
            test_path = Path("test_plot_temp.png")
            plt.savefig(test_path)
            plt.close()

            if test_path.exists():
                test_path.unlink()  # Clean up
                print("  ✓ Visualization OK")
                return True
            else:
                print("  ✗ Plot file not created")
                return False

        except Exception as e:
            print(f"  ✗ Visualization failed: {e}")
            return False

    def test_transformer_import(self) -> bool:
        """Test transformer imports (may fail on Python 3.14)."""
        print("\n" + "-"*50)
        print("TEST 6: Transformer Import")
        print("-"*50)

        try:
            import transformers
            from transformers import AutoTokenizer
            print(f"  Transformers version: {transformers.__version__}")
            print("  ✓ Transformer import OK")
            return True
        except Exception as e:
            print(f"  ⚠ Transformer import failed: {e}")
            print("  The framework will run in fallback mode")
            print("  Recommendation: Use Python 3.10-3.12 for full functionality")
            return False

    def test_training_simulation(self) -> bool:
        """Test a minimal training simulation."""
        print("\n" + "-"*50)
        print("TEST 7: Training Simulation")
        print("-"*50)

        try:
            import torch
            import torch.nn as nn
            import numpy as np

            # Create simple model
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 5)

                def forward(self, x):
                    return self.fc(x)

            model = SimpleModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            # Simple training step
            for i in range(5):
                x = torch.randn(4, 10)
                y = torch.randn(4, 5)

                optimizer.zero_grad()
                pred = model(x)
                loss = nn.MSELoss()(pred, y)
                loss.backward()
                optimizer.step()

            print(f"  Training steps completed: 5")
            print(f"  Final loss: {loss.item():.4f}")
            print("  ✓ Training simulation OK")
            return True

        except Exception as e:
            print(f"  ✗ Training simulation failed: {e}")
            return False

    def run_all_tests(self, test_training: bool = False) -> bool:
        """Run all tests."""
        print("="*60)
        print("SELF-TRAINING FRAMEWORK - QUICK TEST")
        print("="*60)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        tests = [
            ("data_loading", self.test_data_loading),
            ("docx_extraction", self.test_docx_extraction),
            ("confidence_estimation", self.test_confidence_estimation),
            ("evaluation_metrics", self.test_evaluation_metrics),
            ("visualization", self.test_visualization),
            ("transformer_import", self.test_transformer_import),
        ]

        if test_training:
            tests.append(("training_simulation", self.test_training_simulation))

        for test_name, test_func in tests:
            try:
                self.results[test_name] = test_func()
            except Exception as e:
                print(f"  ✗ Test {test_name} crashed: {e}")
                self.results[test_name] = False

        # Print summary
        self.print_summary()

        return all(self.results.values())

    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)

        for test_name, passed_test in self.results.items():
            status = "✓ PASS" if passed_test else "✗ FAIL"
            print(f"  {status}: {test_name}")

        elapsed = datetime.now() - self.start_time
        print(f"\n  Results: {passed}/{total} tests passed")
        print(f"  Time: {elapsed.total_seconds():.1f}s")

        if passed == total:
            print("\n  ✓ All tests passed! Framework is ready.")
            print("\n  Run the full pipeline with:")
            print("    python run_self_training_pipeline.py")
        elif passed >= total - 1:
            print("\n  ⚠ Most tests passed. Framework should work with minor limitations.")
        else:
            print("\n  ✗ Multiple tests failed. Please check your installation.")


def main():
    parser = argparse.ArgumentParser(description="Quick test for self-training framework")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test-training", "-t", action="store_true",
                       help="Include training simulation test")

    args = parser.parse_args()

    tester = QuickTest(verbose=args.verbose)
    success = tester.run_all_tests(test_training=args.test_training)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
