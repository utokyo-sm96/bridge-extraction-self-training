# Self-Training Framework for Bridge Design Information Extraction

A self-training framework that uses pseudo-labels from LLM extractions to train a transformer-based model for extracting structured bridge design parameters from Japanese engineering documents.

## Overview

This framework implements a self-training approach where:
1. Initial pseudo-labels are generated from LLM-based extraction (JSON files)
2. Confidence scores are estimated using source verification, engineering plausibility, and cross-field consistency
3. High-confidence samples are used to train a Japanese BERT model (cl-tohoku/bert-base-japanese-v3)
4. The model iteratively refines labels with progressive threshold relaxation
5. Cross-validation provides statistically rigorous evaluation metrics

## Features

- **Self-Training with Confidence Estimation**: 4-factor scoring (source verification, plausibility, consistency, agreement)
- **Japanese BERT Backbone**: Uses `cl-tohoku/bert-base-japanese-v3` for Japanese text understanding
- **Curriculum Learning**: Progressive confidence threshold relaxation (0.8 → 0.76 → ...)
- **K-Fold Cross-Validation**: For robust evaluation with 95% confidence intervals
- **Baseline Comparison**: Compare against rule-based, statistical, and LLM baselines
- **Publication-Ready Visualizations**: Generate figures for academic papers

## Requirements

- Python 3.9+ (recommended: 3.10-3.12)
- PyTorch with CUDA support (for GPU acceleration)
- 8GB+ GPU memory recommended

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bridge-extraction-self-training.git
cd bridge-extraction-self-training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

Place your data in the following structure:
```
data/
├── documents/     # Source DOCX or TXT files
│   ├── doc001.docx
│   ├── doc002.docx
│   └── ...
└── json/          # Extracted JSON files (pseudo-labels)
    ├── doc001.json
    ├── doc002.json
    └── ...
```

The JSON files should match the document filenames (without extension).

## Quick Start

```bash
# Check dependencies and data
python check_dependencies.py

# Run quick test (2 iterations, 2 folds)
python run_self_training_pipeline.py --quick

# Run full pipeline (5 iterations, 5 folds)
python run_self_training_pipeline.py
```

## Usage

### Basic Usage

```bash
# Full pipeline with default settings
python run_self_training_pipeline.py \
    --documents-dir data/documents \
    --json-dir data/json \
    --output-dir output
```

### Advanced Options

```bash
python run_self_training_pipeline.py \
    --documents-dir data/documents \
    --json-dir data/json \
    --output-dir output \
    --num-iterations 5 \
    --confidence-threshold 0.8 \
    --k-folds 5 \
    --run-llm  # Include LLM baselines (requires OpenAI API key)
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--documents-dir` | `documents` | Directory containing source documents |
| `--json-dir` | `final_json` | Directory containing JSON extractions |
| `--output-dir` | `self_training_output` | Output directory |
| `--num-iterations` | 5 | Number of self-training iterations |
| `--confidence-threshold` | 0.8 | Initial confidence threshold |
| `--k-folds` | 5 | Number of cross-validation folds |
| `--quick` | - | Quick mode (2 iterations, 2 folds) |
| `--skip-training` | - | Skip training, use existing results |
| `--skip-visualization` | - | Skip visualization generation |
| `--run-llm` | - | Include LLM baselines |

## Output

The pipeline generates:

```
output/
├── self_training_results.json      # Complete results
├── refined_labels/                 # Refined JSON labels
│   ├── doc001.json
│   └── ...
└── models/                         # Trained model checkpoints
    └── best_model.pt

plots/
└── self_training/
    ├── confidence_evolution.pdf
    ├── field_accuracy.pdf
    ├── label_changes.pdf
    └── cross_validation.pdf

evaluation_output/
└── evaluation_report.txt

comparison_output/
├── comparison_report.txt
└── comparison_table.tex
```

## GPU Acceleration

The framework automatically detects and uses:
- NVIDIA CUDA GPUs
- Apple Metal (MPS) on M1/M2/M3 Macs
- Falls back to CPU if no GPU available

For best performance on a GPU server:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run with GPU
CUDA_VISIBLE_DEVICES=0 python run_self_training_pipeline.py
```

## Extracted Fields

The framework extracts 30 bridge design parameters including:

| Category | Fields |
|----------|--------|
| **Bridge Type** | bridge_type |
| **Geometry** | girder_length, num_girders, spacing, flange dimensions, web dimensions |
| **Deck** | length, width, thickness |
| **Materials** | concrete_density, steel_density, young_modulus |
| **Loads** | p1_bending, p1_shear, impact_coefficient |
| **Crossbeams** | use_crossbeams, height, thickness, spacing |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Self-Training Loop                        │
├─────────────────────────────────────────────────────────────┤
│  1. Load documents + pseudo-labels                          │
│  2. Estimate confidence scores                              │
│  3. Select high-confidence samples                          │
│  4. Train BERT extraction model                             │
│  5. Generate predictions for all documents                  │
│  6. Merge predictions with pseudo-labels (confidence-based) │
│  7. Lower threshold, repeat from step 3                     │
└─────────────────────────────────────────────────────────────┘
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{bridge_extraction_self_training,
  title={Self-Training Framework for Bridge Design Information Extraction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/bridge-extraction-self-training}
}
```

## License

MIT License - see LICENSE file for details.
