#!/usr/bin/env python3
"""
Self-Training Pipeline Runner
==============================

A convenient script to run the complete self-training pipeline
with sensible defaults and progress tracking.

Usage:
    python run_self_training_pipeline.py                    # Run full pipeline
    python run_self_training_pipeline.py --quick            # Quick test run
    python run_self_training_pipeline.py --eval-only        # Evaluation only
    python run_self_training_pipeline.py --help             # Show options
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)

    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
    }

    optional = {
        'seaborn': 'Seaborn (for enhanced plots)',
        'scipy': 'SciPy (for statistics)',
        'openai': 'OpenAI (for LLM baselines)',
    }

    missing_required = []
    missing_optional = []

    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (REQUIRED)")
            missing_required.append(module)

    print("\nOptional:")
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ○ {name} (optional)")
            missing_optional.append(module)

    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install -r requirements.txt")
        return False

    # Check GPU availability
    import torch
    print("\nCompute Device:")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  ✓ Apple MPS (Metal)")
    else:
        print("  ○ CPU only (training will be slower)")

    return True


def check_data(documents_dir: str, json_dir: str):
    """Check if required data exists."""
    print("\n" + "="*60)
    print("CHECKING DATA")
    print("="*60)

    docs_path = Path(documents_dir)
    json_path = Path(json_dir)

    # Check directories exist
    if not docs_path.exists():
        print(f"  ✗ Documents directory not found: {documents_dir}")
        return False, 0

    if not json_path.exists():
        print(f"  ✗ JSON directory not found: {json_dir}")
        return False, 0

    # Count files
    txt_files = list(docs_path.glob("*.txt"))
    docx_files = list(docs_path.glob("*.docx"))
    json_files = list(json_path.glob("*.json"))

    print(f"  Documents directory: {documents_dir}")
    print(f"    - Text files: {len(txt_files)}")
    print(f"    - DOCX files: {len(docx_files)}")

    print(f"  JSON directory: {json_dir}")
    print(f"    - JSON files: {len(json_files)}")

    # Count matching pairs (TXT or DOCX)
    txt_stems = {f.stem for f in txt_files}
    docx_stems = {f.stem for f in docx_files}
    doc_stems = txt_stems | docx_stems  # Union of both
    json_stems = {f.stem for f in json_files}
    matching = doc_stems & json_stems

    print(f"  Matching pairs: {len(matching)} (TXT: {len(txt_stems & json_stems)}, DOCX: {len(docx_stems & json_stems)})")

    if len(matching) < 10:
        print(f"\n⚠ Warning: Only {len(matching)} matching document-label pairs found.")
        print("  Need at least 10 pairs for meaningful self-training.")
        if len(matching) == 0:
            return False, 0

    return True, len(matching)


def run_self_training(
    documents_dir: str,
    json_dir: str,
    output_dir: str,
    num_iterations: int = 5,
    confidence_threshold: float = 0.8,
    k_folds: int = 5,
    quick_mode: bool = False
):
    """Run the self-training framework."""
    print("\n" + "="*60)
    print("RUNNING SELF-TRAINING")
    print("="*60)

    from self_training_framework import SelfTrainingFramework, SelfTrainingConfig

    # Adjust settings for quick mode
    if quick_mode:
        print("  Mode: QUICK (reduced iterations for testing)")
        num_iterations = 2
        k_folds = 2
    else:
        print("  Mode: FULL")

    print(f"  Iterations: {num_iterations}")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Cross-validation folds: {k_folds}")
    print(f"  Output: {output_dir}")

    # Create configuration
    config = SelfTrainingConfig(
        documents_dir=documents_dir,
        json_dir=json_dir,
        output_dir=output_dir,
        num_iterations=num_iterations,
        initial_confidence_threshold=confidence_threshold,
        k_folds=k_folds,
    )

    if quick_mode:
        config.num_epochs = 3
        config.batch_size = 4

    # Run framework
    framework = SelfTrainingFramework(config)
    results = framework.run_full_pipeline()

    return results


def run_visualization(results_path: str, output_dir: str):
    """Generate visualizations."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    if not Path(results_path).exists():
        print(f"  ✗ Results file not found: {results_path}")
        return False

    try:
        from self_training_visualization import SelfTrainingVisualizer

        visualizer = SelfTrainingVisualizer(results_path, output_dir)
        visualizer.generate_all_figures()

        print(f"  ✓ Figures saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
        return False


def run_evaluation(predictions_dir: str, ground_truth_dir: str, output_dir: str):
    """Run comprehensive evaluation."""
    print("\n" + "="*60)
    print("RUNNING EVALUATION")
    print("="*60)

    try:
        from evaluation_metrics import run_comprehensive_evaluation

        results = run_comprehensive_evaluation(
            predictions_dir,
            ground_truth_dir,
            output_dir
        )

        print(f"  ✓ Evaluation complete")
        print(f"  ✓ Results saved to: {output_dir}")

        return results

    except Exception as e:
        print(f"  ✗ Evaluation failed: {e}")
        logger.exception("Evaluation error")
        return None


def run_baseline_comparison(
    documents_dir: str,
    ground_truth_dir: str,
    output_dir: str,
    max_docs: int = 50,
    run_llm: bool = False
):
    """Run baseline comparison."""
    print("\n" + "="*60)
    print("RUNNING BASELINE COMPARISON")
    print("="*60)

    try:
        from baseline_comparison import BaselineComparison

        # Load documents
        docs_path = Path(documents_dir)
        truth_path = Path(ground_truth_dir)

        documents = []
        ground_truth = []

        for json_file in sorted(truth_path.glob("*.json"))[:max_docs]:
            doc_id = json_file.stem
            txt_file = docs_path / f"{doc_id}.txt"

            if not txt_file.exists():
                continue

            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()

            with open(json_file, 'r', encoding='utf-8') as f:
                truth = json.load(f)

            documents.append((doc_id, text[:10000]))
            ground_truth.append(truth)

        print(f"  Documents: {len(documents)}")
        print(f"  Run LLM baselines: {run_llm}")

        # Run comparison
        comparison = BaselineComparison(output_dir=output_dir)
        comparison.register_all_baselines()

        results = comparison.run_comparison(documents, ground_truth, run_llm=run_llm)

        # Generate reports
        report = comparison.generate_comparison_report()
        comparison.generate_latex_table()
        comparison.save_results()

        print(f"  ✓ Comparison complete")
        print(f"  ✓ Results saved to: {output_dir}")

        return results

    except Exception as e:
        print(f"  ✗ Baseline comparison failed: {e}")
        logger.exception("Comparison error")
        return None


def print_summary(results: dict, start_time: datetime):
    """Print final summary."""
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)

    elapsed = datetime.now() - start_time
    print(f"\nTotal time: {elapsed}")

    if results and "summary" in results:
        summary = results["summary"]
        print(f"\nKey Results:")
        print(f"  Initial confidence: {summary.get('initial_pseudo_label_confidence', 'N/A'):.3f}")
        print(f"  Final confidence:   {summary.get('final_refined_confidence', 'N/A'):.3f}")
        print(f"  Improvement:        +{summary.get('confidence_improvement', 'N/A'):.3f}")

        if 'overall_accuracy_mean' in summary:
            print(f"\nCross-Validation:")
            print(f"  Overall accuracy:   {summary['overall_accuracy_mean']:.3f}")
            if summary.get('overall_accuracy_ci'):
                ci = summary['overall_accuracy_ci']
                print(f"  95% CI:             [{ci[0]:.3f}, {ci[1]:.3f}]")

    print("\nOutput files:")
    print("  - self_training_output/self_training_results.json")
    print("  - plots/self_training/*.pdf")
    print("  - evaluation_output/evaluation_report.txt")
    print("  - comparison_output/comparison_report.txt")

    print("\nNext steps:")
    print("  1. Review visualizations in plots/self_training/")
    print("  2. Check evaluation_report.txt for detailed metrics")
    print("  3. Use comparison_table.tex in your paper")
    print("  4. See paper_methodology_draft.md for paper template")


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete self-training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_self_training_pipeline.py                    # Full pipeline
  python run_self_training_pipeline.py --quick            # Quick test
  python run_self_training_pipeline.py --eval-only        # Evaluation only
  python run_self_training_pipeline.py --skip-training    # Skip training
        """
    )

    # Input/output paths
    parser.add_argument("--documents-dir", type=str, default="data/documents",
                       help="Directory containing source documents")
    parser.add_argument("--json-dir", type=str, default="data/json",
                       help="Directory containing initial JSON extractions")
    parser.add_argument("--output-dir", type=str, default="self_training_output",
                       help="Output directory for results")

    # Self-training options
    parser.add_argument("--num-iterations", type=int, default=5,
                       help="Number of self-training iterations")
    parser.add_argument("--confidence-threshold", type=float, default=0.8,
                       help="Initial confidence threshold")
    parser.add_argument("--k-folds", type=int, default=5,
                       help="Number of cross-validation folds")

    # Mode options
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode (reduced iterations for testing)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip self-training (use existing results)")
    parser.add_argument("--skip-visualization", action="store_true",
                       help="Skip visualization generation")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation")
    parser.add_argument("--skip-comparison", action="store_true",
                       help="Skip baseline comparison")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation (no training)")

    # Baseline options
    parser.add_argument("--run-llm", action="store_true",
                       help="Include LLM baselines in comparison")
    parser.add_argument("--max-comparison-docs", type=int, default=50,
                       help="Maximum documents for baseline comparison")

    args = parser.parse_args()

    # Handle eval-only mode
    if args.eval_only:
        args.skip_training = True
        args.skip_visualization = True
        args.skip_comparison = True

    start_time = datetime.now()

    print("\n" + "="*60)
    print("SELF-TRAINING PIPELINE FOR BRIDGE DESIGN EXTRACTION")
    print("="*60)
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        sys.exit(1)

    # Step 2: Check data
    data_ok, num_pairs = check_data(args.documents_dir, args.json_dir)
    if not data_ok:
        print("\n❌ Please ensure data is available and try again.")
        sys.exit(1)

    results = None

    # Step 3: Run self-training
    if not args.skip_training:
        try:
            results = run_self_training(
                documents_dir=args.documents_dir,
                json_dir=args.json_dir,
                output_dir=args.output_dir,
                num_iterations=args.num_iterations,
                confidence_threshold=args.confidence_threshold,
                k_folds=args.k_folds,
                quick_mode=args.quick
            )
        except Exception as e:
            print(f"\n❌ Self-training failed: {e}")
            logger.exception("Self-training error")
            sys.exit(1)
    else:
        print("\n[SKIPPED] Self-training")
        # Try to load existing results
        results_path = Path(args.output_dir) / "self_training_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)

    # Step 4: Generate visualizations
    if not args.skip_visualization:
        results_path = Path(args.output_dir) / "self_training_results.json"
        run_visualization(
            str(results_path),
            "plots/self_training"
        )
    else:
        print("\n[SKIPPED] Visualization")

    # Step 5: Run evaluation
    if not args.skip_evaluation:
        run_evaluation(
            predictions_dir=args.json_dir,  # Use refined or original
            ground_truth_dir=args.json_dir,
            output_dir="evaluation_output"
        )
    else:
        print("\n[SKIPPED] Evaluation")

    # Step 6: Run baseline comparison
    if not args.skip_comparison:
        run_baseline_comparison(
            documents_dir=args.documents_dir,
            ground_truth_dir=args.json_dir,
            output_dir="comparison_output",
            max_docs=args.max_comparison_docs,
            run_llm=args.run_llm
        )
    else:
        print("\n[SKIPPED] Baseline comparison")

    # Print summary
    print_summary(results, start_time)


if __name__ == "__main__":
    main()
