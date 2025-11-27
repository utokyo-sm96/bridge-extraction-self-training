"""
Visualization Module for Self-Training Framework
=================================================

Generates publication-quality figures for journal paper defense:
1. Training convergence curves
2. Confidence distribution analysis
3. Uncertainty visualization
4. Cross-validation results
5. Field-level performance heatmaps

Author: Self-Training IE Framework
License: MIT
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

# Try to import seaborn for enhanced plots
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    warnings.warn("Seaborn not installed. Some visualizations will be simplified.")

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette for consistent styling
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'tertiary': '#F18F01',
    'success': '#2ECC71',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'neutral': '#95A5A6',
    'dark': '#2C3E50',
}


class SelfTrainingVisualizer:
    """Generates publication-quality visualizations for self-training results."""

    def __init__(self, results_path: str, output_dir: str = "plots/self_training"):
        """
        Initialize visualizer.

        Args:
            results_path: Path to self_training_results.json
            output_dir: Directory for saving plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

    def generate_all_figures(self):
        """Generate all publication figures."""
        print("Generating publication figures...")

        self.plot_confidence_evolution()
        self.plot_training_convergence()
        self.plot_confidence_distribution()
        self.plot_cross_validation_results()
        self.plot_field_performance_heatmap()
        self.plot_uncertainty_analysis()
        self.plot_self_training_workflow()
        self.plot_summary_dashboard()

        print(f"All figures saved to: {self.output_dir}")

    def plot_confidence_evolution(self):
        """Plot confidence improvement across self-training iterations."""
        fig, ax = plt.subplots(figsize=(8, 5))

        history = self.results.get("training_history", [])
        if not history:
            return

        iterations = [h["iteration"] for h in history]
        avg_conf = [h["metrics"]["avg_confidence"] for h in history]
        min_conf = [h["metrics"]["min_confidence"] for h in history]
        max_conf = [h["metrics"]["max_confidence"] for h in history]

        # Plot with confidence band
        ax.fill_between(iterations, min_conf, max_conf, alpha=0.2, color=COLORS['primary'])
        ax.plot(iterations, avg_conf, 'o-', color=COLORS['primary'],
                linewidth=2, markersize=8, label='Average Confidence')
        ax.plot(iterations, min_conf, '--', color=COLORS['secondary'],
                linewidth=1, alpha=0.7, label='Min Confidence')
        ax.plot(iterations, max_conf, '--', color=COLORS['success'],
                linewidth=1, alpha=0.7, label='Max Confidence')

        # Add threshold line
        initial_threshold = history[0].get("threshold", 0.8) if history else 0.8
        ax.axhline(y=initial_threshold, color=COLORS['warning'], linestyle=':',
                   linewidth=2, label=f'Initial Threshold ({initial_threshold})')

        ax.set_xlabel('Self-Training Iteration')
        ax.set_ylabel('Confidence Score')
        ax.set_title('Pseudo-Label Confidence Evolution During Self-Training')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(iterations)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_evolution.pdf')
        plt.savefig(self.output_dir / 'confidence_evolution.png')
        plt.close()

    def plot_training_convergence(self):
        """Plot training convergence metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        history = self.results.get("training_history", [])
        if not history:
            return

        iterations = [h["iteration"] for h in history]

        # Left: Number of high-confidence samples
        num_samples = [h["num_samples"] for h in history]
        axes[0].bar(iterations, num_samples, color=COLORS['primary'], alpha=0.8)
        axes[0].set_xlabel('Self-Training Iteration')
        axes[0].set_ylabel('Number of High-Confidence Samples')
        axes[0].set_title('Training Sample Selection')
        axes[0].set_xticks(iterations)

        # Add trend line
        z = np.polyfit(iterations, num_samples, 1)
        p = np.poly1d(z)
        axes[0].plot(iterations, p(iterations), '--', color=COLORS['danger'],
                     linewidth=2, label='Trend')
        axes[0].legend()

        # Right: Label change rate
        change_rates = [h["metrics"]["change_rate"] for h in history]
        axes[1].plot(iterations, change_rates, 's-', color=COLORS['secondary'],
                     linewidth=2, markersize=8)
        axes[1].set_xlabel('Self-Training Iteration')
        axes[1].set_ylabel('Label Change Rate')
        axes[1].set_title('Label Refinement Rate')
        axes[1].set_xticks(iterations)
        axes[1].set_ylim(0, max(change_rates) * 1.2 if change_rates else 0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_convergence.pdf')
        plt.savefig(self.output_dir / 'training_convergence.png')
        plt.close()

    def plot_confidence_distribution(self):
        """Plot initial vs final confidence distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        initial = self.results.get("initial_confidence", {}).get("per_document", [])
        final = self.results.get("final_confidences", [])

        if not initial or not final:
            return

        # Left: Histogram comparison
        bins = np.linspace(0, 1, 21)
        axes[0].hist(initial, bins=bins, alpha=0.6, label='Initial (LLM)',
                     color=COLORS['secondary'], edgecolor='white')
        axes[0].hist(final, bins=bins, alpha=0.6, label='Refined (Self-Training)',
                     color=COLORS['primary'], edgecolor='white')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Number of Documents')
        axes[0].set_title('Confidence Distribution: Before vs After')
        axes[0].legend()

        # Add vertical lines for means
        axes[0].axvline(np.mean(initial), color=COLORS['secondary'],
                        linestyle='--', linewidth=2)
        axes[0].axvline(np.mean(final), color=COLORS['primary'],
                        linestyle='--', linewidth=2)

        # Right: Paired improvement
        if len(initial) == len(final):
            improvements = [f - i for i, f in zip(initial, final)]

            colors = [COLORS['success'] if imp > 0 else COLORS['danger'] for imp in improvements]
            axes[1].bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].set_xlabel('Document Index')
            axes[1].set_ylabel('Confidence Improvement')
            axes[1].set_title('Per-Document Confidence Change')

            # Add summary statistics
            pos_count = sum(1 for imp in improvements if imp > 0)
            textstr = f'Improved: {pos_count}/{len(improvements)} ({100*pos_count/len(improvements):.1f}%)'
            axes[1].text(0.95, 0.95, textstr, transform=axes[1].transAxes,
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_distribution.pdf')
        plt.savefig(self.output_dir / 'confidence_distribution.png')
        plt.close()

    def plot_cross_validation_results(self):
        """Plot cross-validation performance metrics."""
        cv_results = self.results.get("cross_validation", {})
        if not cv_results:
            return

        # Extract field-level results
        field_metrics = {}
        for key, value in cv_results.items():
            if key.endswith("_accuracy") and isinstance(value, dict):
                field_name = key.replace("_accuracy", "")
                field_metrics[field_name] = value

        if not field_metrics:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Box plot of field accuracies
        field_names = list(field_metrics.keys())
        means = [field_metrics[f]["mean"] for f in field_names]
        stds = [field_metrics[f]["std"] for f in field_names]
        ci_lowers = [field_metrics[f]["ci_95_lower"] for f in field_names]
        ci_uppers = [field_metrics[f]["ci_95_upper"] for f in field_names]

        # Sort by mean accuracy
        sorted_indices = np.argsort(means)[::-1]
        field_names = [field_names[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]
        ci_lowers = [ci_lowers[i] for i in sorted_indices]
        ci_uppers = [ci_uppers[i] for i in sorted_indices]

        y_pos = np.arange(len(field_names))

        # Create horizontal bar chart with error bars
        bars = axes[0].barh(y_pos, means, xerr=stds, align='center',
                           color=COLORS['primary'], alpha=0.8, capsize=3)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels([f.replace('_', ' ').title() for f in field_names])
        axes[0].set_xlabel('Accuracy')
        axes[0].set_title('Field-Level Extraction Accuracy (K-Fold CV)')
        axes[0].set_xlim(0, 1.1)

        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            axes[0].text(mean + std + 0.02, i, f'{mean:.2f}±{std:.2f}',
                        va='center', fontsize=8)

        # Right: Overall performance with confidence interval
        overall = cv_results.get("overall_field_accuracy", {})
        if overall:
            # Create forest plot style visualization
            y_positions = [0]
            labels = ['Overall']

            axes[1].errorbar(
                [overall["mean"]],
                y_positions,
                xerr=[[overall["mean"] - overall["ci_95_lower"]],
                      [overall["ci_95_upper"] - overall["mean"]]],
                fmt='D',
                markersize=12,
                color=COLORS['primary'],
                capsize=8,
                capthick=2,
                elinewidth=2
            )

            axes[1].axvline(x=overall["mean"], color=COLORS['primary'],
                           linestyle='--', alpha=0.5)
            axes[1].set_xlim(0, 1.1)
            axes[1].set_yticks(y_positions)
            axes[1].set_yticklabels(labels)
            axes[1].set_xlabel('Accuracy')
            axes[1].set_title('Overall Performance with 95% CI')

            # Add text annotation
            textstr = f'Mean: {overall["mean"]:.3f}\n95% CI: [{overall["ci_95_lower"]:.3f}, {overall["ci_95_upper"]:.3f}]'
            axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes,
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_validation_results.pdf')
        plt.savefig(self.output_dir / 'cross_validation_results.png')
        plt.close()

    def plot_field_performance_heatmap(self):
        """Plot heatmap of field performance across iterations."""
        cv_results = self.results.get("cross_validation", {})
        if not cv_results:
            return

        # Build matrix of field performances
        field_metrics = {}
        for key, value in cv_results.items():
            if key.endswith("_accuracy") and isinstance(value, dict):
                field_name = key.replace("_accuracy", "")
                if "values" in value:
                    field_metrics[field_name] = value["values"]

        if not field_metrics:
            return

        # Create matrix
        field_names = list(field_metrics.keys())
        n_folds = len(list(field_metrics.values())[0])

        matrix = np.zeros((len(field_names), n_folds))
        for i, field in enumerate(field_names):
            matrix[i, :] = field_metrics[field]

        fig, ax = plt.subplots(figsize=(10, 8))

        if HAS_SEABORN:
            sns.heatmap(
                matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                xticklabels=[f'Fold {i+1}' for i in range(n_folds)],
                yticklabels=[f.replace('_', ' ').title() for f in field_names],
                ax=ax,
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Accuracy'}
            )
        else:
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(range(n_folds))
            ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
            ax.set_yticks(range(len(field_names)))
            ax.set_yticklabels([f.replace('_', ' ').title() for f in field_names])
            plt.colorbar(im, ax=ax, label='Accuracy')

            # Add text annotations
            for i in range(len(field_names)):
                for j in range(n_folds):
                    ax.text(j, i, f'{matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=8)

        ax.set_title('Field Performance Across Cross-Validation Folds')
        ax.set_xlabel('CV Fold')
        ax.set_ylabel('Extraction Field')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'field_performance_heatmap.pdf')
        plt.savefig(self.output_dir / 'field_performance_heatmap.png')
        plt.close()

    def plot_uncertainty_analysis(self):
        """Plot uncertainty analysis for sampled documents."""
        uncertainty_samples = self.results.get("uncertainty_samples", {})
        if not uncertainty_samples:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Collect uncertainty data
        all_cvs = []
        all_stds = []
        field_names = []

        for doc_id, data in uncertainty_samples.items():
            field_uncert = data.get("field_uncertainties", {})
            for field, metrics in field_uncert.items():
                all_cvs.append(metrics.get("cv", 0))
                all_stds.append(metrics.get("std", 0))
                if field not in field_names:
                    field_names.append(field)

        # Left: Coefficient of variation distribution
        axes[0].hist(all_cvs, bins=20, color=COLORS['secondary'], alpha=0.8, edgecolor='white')
        axes[0].axvline(np.mean(all_cvs), color=COLORS['danger'],
                        linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_cvs):.3f}')
        axes[0].set_xlabel('Coefficient of Variation (CV)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Prediction Uncertainty Distribution')
        axes[0].legend()

        # Right: Reliability scores per document
        reliabilities = []
        doc_ids = []
        for doc_id, data in uncertainty_samples.items():
            rel = data.get("reliability", {}).get("overall_reliability", 0)
            reliabilities.append(rel)
            doc_ids.append(doc_id[:15])  # Truncate for display

        sorted_indices = np.argsort(reliabilities)[::-1]
        reliabilities = [reliabilities[i] for i in sorted_indices]
        doc_ids = [doc_ids[i] for i in sorted_indices]

        colors = [COLORS['success'] if r > 0.7 else
                  COLORS['warning'] if r > 0.5 else COLORS['danger']
                  for r in reliabilities]

        axes[1].barh(range(len(doc_ids)), reliabilities, color=colors, alpha=0.8)
        axes[1].set_yticks(range(len(doc_ids)))
        axes[1].set_yticklabels(doc_ids)
        axes[1].set_xlabel('Reliability Score')
        axes[1].set_title('Document-Level Reliability')
        axes[1].set_xlim(0, 1.1)

        # Add threshold lines
        axes[1].axvline(0.7, color=COLORS['success'], linestyle='--', alpha=0.7, label='High (>0.7)')
        axes[1].axvline(0.5, color=COLORS['warning'], linestyle='--', alpha=0.7, label='Medium (>0.5)')
        axes[1].legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'uncertainty_analysis.pdf')
        plt.savefig(self.output_dir / 'uncertainty_analysis.png')
        plt.close()

    def plot_self_training_workflow(self):
        """Create a visual diagram of the self-training workflow."""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')

        # Define box positions and connections
        boxes = [
            {"pos": (1, 6), "text": "DOCX\nDocuments", "color": COLORS['neutral']},
            {"pos": (4, 6), "text": "LLM\nExtraction\n(GPT-4)", "color": COLORS['tertiary']},
            {"pos": (7, 6), "text": "Initial\nPseudo-Labels", "color": COLORS['warning']},
            {"pos": (10, 6), "text": "Confidence\nEstimation", "color": COLORS['secondary']},
            {"pos": (10, 4), "text": "High-Confidence\nFiltering", "color": COLORS['primary']},
            {"pos": (7, 4), "text": "Transformer\nModel Training", "color": COLORS['success']},
            {"pos": (4, 4), "text": "Prediction\nGeneration", "color": COLORS['primary']},
            {"pos": (4, 2), "text": "Label\nRefinement", "color": COLORS['secondary']},
            {"pos": (7, 2), "text": "Uncertainty\nQuantification", "color": COLORS['tertiary']},
            {"pos": (10, 2), "text": "Cross-\nValidation", "color": COLORS['success']},
            {"pos": (13, 4), "text": "Refined\nJSON Output", "color": COLORS['success']},
        ]

        # Draw boxes
        for box in boxes:
            x, y = box["pos"]
            rect = mpatches.FancyBboxPatch(
                (x - 0.9, y - 0.5), 1.8, 1,
                boxstyle="round,pad=0.1",
                facecolor=box["color"],
                edgecolor='white',
                linewidth=2,
                alpha=0.9
            )
            ax.add_patch(rect)
            ax.text(x, y, box["text"], ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')

        # Draw arrows
        arrows = [
            ((1.9, 6), (3.1, 6)),  # DOCX -> LLM
            ((4.9, 6), (6.1, 6)),  # LLM -> Pseudo-Labels
            ((7.9, 6), (9.1, 6)),  # Pseudo-Labels -> Confidence
            ((10, 5.5), (10, 4.5)),  # Confidence -> Filtering
            ((9.1, 4), (7.9, 4)),  # Filtering -> Training
            ((6.1, 4), (4.9, 4)),  # Training -> Prediction
            ((4, 3.5), (4, 2.5)),  # Prediction -> Refinement
            ((4.9, 2), (6.1, 2)),  # Refinement -> Uncertainty
            ((7.9, 2), (9.1, 2)),  # Uncertainty -> CV
            ((10.9, 4), (12.1, 4)),  # -> Output
        ]

        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                       lw=2, connectionstyle='arc3,rad=0'))

        # Add self-training loop arrow
        ax.annotate('', xy=(7, 5.5), xytext=(5, 2.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['danger'],
                                   lw=2, connectionstyle='arc3,rad=-0.3',
                                   linestyle='dashed'))
        ax.text(5.5, 4.2, 'Iterate', fontsize=9, color=COLORS['danger'],
               fontweight='bold', rotation=45)

        # Title
        ax.text(7, 7.5, 'Self-Training Framework for Bridge Design Information Extraction',
               ha='center', va='center', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'self_training_workflow.pdf')
        plt.savefig(self.output_dir / 'self_training_workflow.png')
        plt.close()

    def plot_summary_dashboard(self):
        """Create a summary dashboard with key metrics."""
        summary = self.results.get("summary", {})
        if not summary:
            return

        fig = plt.figure(figsize=(16, 10))

        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Key Metrics Panel (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')

        metrics_text = f"""
        SELF-TRAINING RESULTS SUMMARY
        ══════════════════════════════

        Methodology: {summary.get('methodology', 'N/A')}
        Architecture: {summary.get('model_architecture', 'N/A')}

        CONFIDENCE SCORES
        ─────────────────
        Initial: {summary.get('initial_pseudo_label_confidence', 0):.3f}
        Final:   {summary.get('final_refined_confidence', 0):.3f}
        Δ:       +{summary.get('confidence_improvement', 0):.3f}

        TRAINING INFO
        ─────────────
        Iterations: {summary.get('num_self_training_iterations', 0)}
        CV Folds:   {summary.get('cross_validation_folds', 0)}
        """
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                fontsize=10, family='monospace', verticalalignment='top')

        # 2. Confidence Improvement Gauge (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        initial = summary.get('initial_pseudo_label_confidence', 0)
        final = summary.get('final_refined_confidence', 0)

        # Simple bar comparison
        categories = ['Initial\n(LLM)', 'Final\n(Self-Trained)']
        values = [initial, final]
        colors_bar = [COLORS['secondary'], COLORS['success']]

        bars = ax2.bar(categories, values, color=colors_bar, alpha=0.8, edgecolor='white', linewidth=2)
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence Improvement')
        ax2.set_ylim(0, 1.1)

        # Add value labels
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

        # Add improvement arrow
        if final > initial:
            ax2.annotate('', xy=(1, final), xytext=(0, initial),
                        arrowprops=dict(arrowstyle='->', color=COLORS['success'],
                                       lw=3, connectionstyle='arc3,rad=0.2'))
            improvement_pct = (final - initial) / initial * 100 if initial > 0 else 0
            ax2.text(0.5, (initial + final)/2, f'+{improvement_pct:.1f}%',
                    ha='center', fontsize=12, fontweight='bold', color=COLORS['success'])

        # 3. Defensibility Checklist (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')

        defense_metrics = summary.get('defensibility_metrics', {})

        checklist_items = [
            ('Cross-Validated', defense_metrics.get('cross_validated', False)),
            ('Uncertainty Quantified', defense_metrics.get('uncertainty_quantified', False)),
            ('Confidence Intervals', defense_metrics.get('confidence_interval_provided', False)),
            ('Multi-Source Verification', defense_metrics.get('multi_source_verification', False)),
        ]

        y_start = 0.9
        for i, (item, checked) in enumerate(checklist_items):
            symbol = '✓' if checked else '✗'
            color = COLORS['success'] if checked else COLORS['danger']
            ax3.text(0.1, y_start - i*0.2, symbol, fontsize=20, color=color,
                    transform=ax3.transAxes)
            ax3.text(0.25, y_start - i*0.2, item, fontsize=11,
                    transform=ax3.transAxes, verticalalignment='center')

        ax3.text(0.5, 1.0, 'Defensibility Checklist', transform=ax3.transAxes,
                ha='center', fontsize=12, fontweight='bold')

        # 4. Training History (bottom spanning)
        ax4 = fig.add_subplot(gs[1:, :])

        history = self.results.get("training_history", [])
        if history:
            iterations = [h["iteration"] for h in history]
            avg_conf = [h["metrics"]["avg_confidence"] for h in history]
            num_samples = [h["num_samples"] for h in history]
            change_rates = [h["metrics"]["change_rate"] * 100 for h in history]

            ax4_twin1 = ax4.twinx()
            ax4_twin2 = ax4.twinx()
            ax4_twin2.spines['right'].set_position(('outward', 60))

            p1, = ax4.plot(iterations, avg_conf, 'o-', color=COLORS['primary'],
                          linewidth=2, markersize=8, label='Avg Confidence')
            p2, = ax4_twin1.plot(iterations, num_samples, 's--', color=COLORS['secondary'],
                                linewidth=2, markersize=8, label='# Samples')
            p3, = ax4_twin2.plot(iterations, change_rates, '^:', color=COLORS['tertiary'],
                                linewidth=2, markersize=8, label='Change Rate %')

            ax4.set_xlabel('Self-Training Iteration')
            ax4.set_ylabel('Confidence', color=COLORS['primary'])
            ax4_twin1.set_ylabel('Samples', color=COLORS['secondary'])
            ax4_twin2.set_ylabel('Change Rate %', color=COLORS['tertiary'])

            ax4.tick_params(axis='y', labelcolor=COLORS['primary'])
            ax4_twin1.tick_params(axis='y', labelcolor=COLORS['secondary'])
            ax4_twin2.tick_params(axis='y', labelcolor=COLORS['tertiary'])

            lines = [p1, p2, p3]
            ax4.legend(lines, [l.get_label() for l in lines], loc='upper right')
            ax4.set_title('Training Progression Over Self-Training Iterations')

        plt.suptitle('Self-Training Framework: Results Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(self.output_dir / 'summary_dashboard.pdf')
        plt.savefig(self.output_dir / 'summary_dashboard.png')
        plt.close()


def main():
    """Main entry point for visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate self-training visualizations")
    parser.add_argument(
        "--results",
        type=str,
        default="self_training_output/self_training_results.json",
        help="Path to self-training results JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/self_training",
        help="Output directory for figures"
    )

    args = parser.parse_args()

    visualizer = SelfTrainingVisualizer(args.results, args.output_dir)
    visualizer.generate_all_figures()


if __name__ == "__main__":
    main()
