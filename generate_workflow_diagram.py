"""
Generate workflow diagram for PointWSSIS evaluation framework.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines


def create_workflow_diagram():
    """Create a visual workflow diagram."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Title
    ax.text(
        5,
        11.5,
        "PointWSSIS Evaluation Framework Workflow",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
    )

    # Define colors
    color_data = "#E8F5E9"  # Light green
    color_config = "#FFF3E0"  # Light orange
    color_eval = "#E3F2FD"  # Light blue
    color_viz = "#F3E5F5"  # Light purple
    color_output = "#FFEBEE"  # Light red

    # Level 1: Input Layer
    y_level1 = 9.5

    # COCO Dataset
    box1 = FancyBboxPatch(
        (0.2, y_level1),
        1.5,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor=color_data,
        edgecolor="green",
        linewidth=2,
    )
    ax.add_patch(box1)
    ax.text(
        0.95,
        y_level1 + 0.4,
        "COCO\nDataset",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Model Weights
    box2 = FancyBboxPatch(
        (2.0, y_level1),
        1.5,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor=color_data,
        edgecolor="green",
        linewidth=2,
    )
    ax.add_patch(box2)
    ax.text(
        2.75,
        y_level1 + 0.4,
        "Model\nWeights",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Configuration
    box3 = FancyBboxPatch(
        (3.8, y_level1),
        1.5,
        0.8,
        boxstyle="round,pad=0.1",
        facecolor=color_config,
        edgecolor="orange",
        linewidth=2,
    )
    ax.add_patch(box3)
    ax.text(
        4.55,
        y_level1 + 0.4,
        "Config\n(YAML)",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    # Level 2: Scripts
    y_level2 = 7.5

    # evaluate_pipeline.py
    box4 = FancyBboxPatch(
        (0.5, y_level2),
        2.5,
        1.0,
        boxstyle="round,pad=0.1",
        facecolor=color_eval,
        edgecolor="blue",
        linewidth=3,
    )
    ax.add_patch(box4)
    ax.text(
        1.75,
        y_level2 + 0.7,
        "evaluate_pipeline.py",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )
    ax.text(
        1.75,
        y_level2 + 0.3,
        "Main Evaluation",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    # evaluation_utils.py
    box5 = FancyBboxPatch(
        (3.5, y_level2),
        2.0,
        1.0,
        boxstyle="round,pad=0.1",
        facecolor=color_eval,
        edgecolor="blue",
        linewidth=2,
    )
    ax.add_patch(box5)
    ax.text(
        4.5,
        y_level2 + 0.7,
        "evaluation_utils.py",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    ax.text(
        4.5,
        y_level2 + 0.3,
        "Helper Functions",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    # run_evaluation.sh
    box6 = FancyBboxPatch(
        (6.0, y_level2),
        2.0,
        1.0,
        boxstyle="round,pad=0.1",
        facecolor="#FFF9C4",
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(box6)
    ax.text(
        7.0,
        y_level2 + 0.7,
        "run_evaluation.sh",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    ax.text(
        7.0,
        y_level2 + 0.3,
        "Automated Script",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Level 3: Processing
    y_level3 = 5.5

    # Metrics Computation
    box7 = FancyBboxPatch(
        (0.5, y_level3),
        1.8,
        1.2,
        boxstyle="round,pad=0.1",
        facecolor="#E0F7FA",
        edgecolor="cyan",
        linewidth=2,
    )
    ax.add_patch(box7)
    ax.text(
        1.4,
        y_level3 + 0.9,
        "Metrics",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    ax.text(
        1.4,
        y_level3 + 0.5,
        "• AP\n• AP50/AP75\n• APs/APm/APl",
        ha="center",
        va="center",
        fontsize=8,
    )

    # Analysis
    box8 = FancyBboxPatch(
        (2.8, y_level3),
        1.8,
        1.2,
        boxstyle="round,pad=0.1",
        facecolor="#E0F7FA",
        edgecolor="cyan",
        linewidth=2,
    )
    ax.add_patch(box8)
    ax.text(
        3.7,
        y_level3 + 0.9,
        "Analysis",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    ax.text(
        3.7,
        y_level3 + 0.5,
        "• Statistics\n• Improvement\n• Efficiency",
        ha="center",
        va="center",
        fontsize=8,
    )

    # Export
    box9 = FancyBboxPatch(
        (5.1, y_level3),
        1.8,
        1.2,
        boxstyle="round,pad=0.1",
        facecolor="#E0F7FA",
        edgecolor="cyan",
        linewidth=2,
    )
    ax.add_patch(box9)
    ax.text(
        6.0,
        y_level3 + 0.9,
        "Export",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    ax.text(
        6.0,
        y_level3 + 0.5,
        "• JSON\n• CSV\n• LaTeX",
        ha="center",
        va="center",
        fontsize=8,
    )

    # Level 4: Visualization
    y_level4 = 3.5

    # visualize_results.py
    box10 = FancyBboxPatch(
        (1.5, y_level4),
        5.0,
        1.0,
        boxstyle="round,pad=0.1",
        facecolor=color_viz,
        edgecolor="purple",
        linewidth=3,
    )
    ax.add_patch(box10)
    ax.text(
        4.0,
        y_level4 + 0.7,
        "visualize_results.py",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )
    ax.text(
        4.0,
        y_level4 + 0.3,
        "Generate 6 Types of Plots",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Level 5: Output Files
    y_level5 = 1.0

    outputs = [
        ("Results\nJSON", 0.5),
        ("Summary\nTable", 1.8),
        ("AP Trends\nPlot", 3.1),
        ("Dashboard\nPlot", 4.4),
        ("CSV\nExport", 5.7),
        ("LaTeX\nTable", 7.0),
    ]

    for label, x_pos in outputs:
        box = FancyBboxPatch(
            (x_pos, y_level5),
            1.0,
            0.8,
            boxstyle="round,pad=0.05",
            facecolor=color_output,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(
            x_pos + 0.5,
            y_level5 + 0.4,
            label,
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
        )

    # Arrows - Level 1 to 2
    arrow_props = dict(arrowstyle="->", lw=2, color="black")

    # Data to eval
    ax.annotate(
        "", xy=(1.75, y_level2 + 1.0), xytext=(0.95, y_level1), arrowprops=arrow_props
    )
    ax.annotate(
        "", xy=(1.75, y_level2 + 1.0), xytext=(2.75, y_level1), arrowprops=arrow_props
    )
    ax.annotate(
        "", xy=(1.75, y_level2 + 1.0), xytext=(4.55, y_level1), arrowprops=arrow_props
    )

    # Level 2 to 3
    ax.annotate(
        "", xy=(1.4, y_level3 + 1.2), xytext=(1.75, y_level2), arrowprops=arrow_props
    )
    ax.annotate(
        "", xy=(3.7, y_level3 + 1.2), xytext=(1.75, y_level2), arrowprops=arrow_props
    )
    ax.annotate(
        "", xy=(6.0, y_level3 + 1.2), xytext=(1.75, y_level2), arrowprops=arrow_props
    )

    # Level 3 to 4
    ax.annotate(
        "", xy=(2.5, y_level4 + 1.0), xytext=(1.4, y_level3), arrowprops=arrow_props
    )
    ax.annotate(
        "", xy=(4.0, y_level4 + 1.0), xytext=(3.7, y_level3), arrowprops=arrow_props
    )
    ax.annotate(
        "", xy=(5.5, y_level4 + 1.0), xytext=(6.0, y_level3), arrowprops=arrow_props
    )

    # Level 4 to 5
    for label, x_pos in outputs:
        ax.annotate(
            "",
            xy=(x_pos + 0.5, y_level5 + 0.8),
            xytext=(4.0, y_level4),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="gray", alpha=0.7),
        )

    # Add legend for weak label types
    legend_y = 10.8
    ax.text(
        7.0,
        legend_y,
        "Weak Label Types:",
        ha="left",
        va="center",
        fontsize=10,
        weight="bold",
    )

    types = [
        "U: Unlabeled",
        "I: Image-level",
        "P: Point (default)",
        "B: Box",
        "F: Full mask",
    ]

    for i, label_type in enumerate(types):
        ax.text(
            7.0,
            legend_y - 0.3 * (i + 1),
            f"• {label_type}",
            ha="left",
            va="center",
            fontsize=8,
        )

    # Add proportions info
    prop_y = 9.2
    ax.text(
        7.0,
        prop_y,
        "Proportions Tested:",
        ha="left",
        va="center",
        fontsize=10,
        weight="bold",
    )

    proportions = ["1%, 2%, 5%, 10%,", "20%, 30%, 50%"]
    for i, prop in enumerate(proportions):
        ax.text(7.0, prop_y - 0.25 * (i + 1), prop, ha="left", va="center", fontsize=8)

    # Add metrics info
    metrics_y = 8.5
    ax.text(
        7.0,
        metrics_y,
        "Metrics Computed:",
        ha="left",
        va="center",
        fontsize=10,
        weight="bold",
    )

    metrics = ["AP, AP50, AP75", "APs, APm, APl", "bbox metrics"]

    for i, metric in enumerate(metrics):
        ax.text(
            7.0,
            metrics_y - 0.25 * (i + 1),
            f"• {metric}",
            ha="left",
            va="center",
            fontsize=8,
        )

    # Add note at bottom
    ax.text(
        5,
        0.2,
        "Complete evaluation pipeline: From raw data to publication-ready results",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="gray",
    )

    plt.tight_layout()

    # Save
    plt.savefig(
        "evaluation_workflow.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("✓ Workflow diagram saved: evaluation_workflow.png")

    return fig


if __name__ == "__main__":
    print("\nGenerating PointWSSIS Evaluation Workflow Diagram...")
    print("=" * 60)
    create_workflow_diagram()
    print("=" * 60)
    print("\nDiagram shows:")
    print("  • Input: Dataset, models, configuration")
    print("  • Processing: Evaluation, metrics, analysis")
    print("  • Visualization: Multiple plot types")
    print("  • Output: JSON, CSV, LaTeX, plots")
    print("\nDone! 🎉\n")
