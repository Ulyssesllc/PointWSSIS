"""
Visualization Module for PointWSSIS Evaluation Results

This script generates comprehensive visualizations for comparing performance
across different weak supervision scenarios.

Generates:
1. AP trend lines across different label proportions
2. Comparison between teacher and student networks
3. Performance improvement analysis
4. Multi-metric comparison plots
5. Weak label type comparison
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


class ResultVisualizer:
    """
    Visualizer for PointWSSIS evaluation results.
    """

    def __init__(self, results_path, output_dir=None):
        """
        Args:
            results_path: Path to evaluation results JSON file
            output_dir: Directory to save generated plots
        """
        self.results_path = results_path
        self.output_dir = output_dir or os.path.dirname(results_path)

        # Load results
        with open(results_path, "r") as f:
            self.results = json.load(f)

        # Parse and organize data
        self.data = self._parse_results()

    def _parse_results(self):
        """
        Parse results into structured format for plotting.

        Returns:
            Dictionary containing organized data
        """
        data = {
            "proportions": [],
            "full_label_percent": [],
            "weak_label_percent": [],
            "weak_types": [],
            "teacher_ap": [],
            "student_ap": [],
            "teacher_ap50": [],
            "student_ap50": [],
            "teacher_ap75": [],
            "student_ap75": [],
            "teacher_aps": [],
            "student_aps": [],
            "teacher_apm": [],
            "student_apm": [],
            "teacher_apl": [],
            "student_apl": [],
            "improvement": [],
        }

        for proportion, result in sorted(
            self.results.items(), key=lambda x: x[1]["full_label_percent"]
        ):
            data["proportions"].append(proportion)
            data["full_label_percent"].append(result["full_label_percent"])
            data["weak_label_percent"].append(result["weak_label_percent"])
            data["weak_types"].append(result["weak_type"])

            # Teacher metrics
            teacher = result.get("teacher_metrics", {})
            data["teacher_ap"].append(teacher.get("AP", 0.0))
            data["teacher_ap50"].append(teacher.get("AP50", 0.0))
            data["teacher_ap75"].append(teacher.get("AP75", 0.0))
            data["teacher_aps"].append(teacher.get("APs", 0.0))
            data["teacher_apm"].append(teacher.get("APm", 0.0))
            data["teacher_apl"].append(teacher.get("APl", 0.0))

            # Student metrics
            student = result.get("student_metrics", {})
            data["student_ap"].append(student.get("AP", 0.0))
            data["student_ap50"].append(student.get("AP50", 0.0))
            data["student_ap75"].append(student.get("AP75", 0.0))
            data["student_aps"].append(student.get("APs", 0.0))
            data["student_apm"].append(student.get("APm", 0.0))
            data["student_apl"].append(student.get("APl", 0.0))

            # Improvement
            improvement = student.get("AP", 0.0) - teacher.get("AP", 0.0)
            data["improvement"].append(improvement)

        return data

    def plot_ap_trends(self, save_name="ap_trends.png"):
        """
        Plot AP trends across different label proportions.

        Args:
            save_name: Filename for saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        x = self.data["full_label_percent"]

        # Plot teacher and student AP
        ax.plot(
            x,
            self.data["teacher_ap"],
            "o-",
            linewidth=2.5,
            markersize=8,
            label="Teacher Network",
            color="#1f77b4",
        )
        ax.plot(
            x,
            self.data["student_ap"],
            "s-",
            linewidth=2.5,
            markersize=8,
            label="Student Network (w/ Weak Labels)",
            color="#ff7f0e",
        )

        # Add improvement annotations
        for i, (xi, teacher_ap, student_ap) in enumerate(
            zip(x, self.data["teacher_ap"], self.data["student_ap"])
        ):
            if student_ap > teacher_ap:
                improvement = student_ap - teacher_ap
                ax.annotate(
                    f"+{improvement:.1f}",
                    xy=(xi, student_ap),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color="green",
                    weight="bold",
                )

        ax.set_xlabel("Full Label Percentage (%)", fontsize=14, weight="bold")
        ax.set_ylabel("Average Precision (AP)", fontsize=14, weight="bold")
        ax.set_title(
            "Instance Segmentation Performance vs Full Label Proportion",
            fontsize=16,
            weight="bold",
            pad=20,
        )

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=12, loc="lower right", framealpha=0.9)

        # Set x-axis to show all proportions
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(p)}%" for p in x])

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        return fig

    def plot_multi_metric_comparison(self, save_name="multi_metric_comparison.png"):
        """
        Plot comparison of multiple AP metrics (AP, AP50, AP75).

        Args:
            save_name: Filename for saved plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Comprehensive Performance Metrics Comparison",
            fontsize=18,
            weight="bold",
            y=1.00,
        )

        x = self.data["full_label_percent"]

        metrics = [
            ("AP", "teacher_ap", "student_ap", "Average Precision"),
            ("AP50", "teacher_ap50", "student_ap50", "AP @ IoU=0.50"),
            ("AP75", "teacher_ap75", "student_ap75", "AP @ IoU=0.75"),
            ("APs", "teacher_aps", "student_aps", "AP Small Objects"),
            ("APm", "teacher_apm", "student_apm", "AP Medium Objects"),
            ("APl", "teacher_apl", "student_apl", "AP Large Objects"),
        ]

        for idx, (metric_name, teacher_key, student_key, title) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]

            teacher_vals = self.data[teacher_key]
            student_vals = self.data[student_key]

            ax.plot(
                x,
                teacher_vals,
                "o-",
                linewidth=2,
                markersize=6,
                label="Teacher",
                alpha=0.7,
            )
            ax.plot(
                x,
                student_vals,
                "s-",
                linewidth=2,
                markersize=6,
                label="Student",
                alpha=0.7,
            )

            ax.set_title(title, fontsize=12, weight="bold")
            ax.set_xlabel("Full Label %", fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(p)}" for p in x], fontsize=8)

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        return fig

    def plot_improvement_analysis(self, save_name="improvement_analysis.png"):
        """
        Plot improvement analysis showing gains from weak supervision.

        Args:
            save_name: Filename for saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        x = self.data["full_label_percent"]
        improvements = self.data["improvement"]

        # Plot 1: Absolute improvement
        colors = ["green" if imp > 0 else "red" for imp in improvements]
        bars = ax1.bar(
            range(len(x)),
            improvements,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{imp:+.2f}",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=10,
                weight="bold",
            )

        ax1.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax1.set_xlabel("Full Label Percentage", fontsize=12, weight="bold")
        ax1.set_ylabel("AP Improvement (Student - Teacher)", fontsize=12, weight="bold")
        ax1.set_title(
            "Performance Improvement with Weak Supervision", fontsize=14, weight="bold"
        )
        ax1.set_xticks(range(len(x)))
        ax1.set_xticklabels([f"{int(p)}%" for p in x])
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Relative improvement percentage
        relative_improvements = [
            (student - teacher) / teacher * 100 if teacher > 0 else 0
            for student, teacher in zip(
                self.data["student_ap"], self.data["teacher_ap"]
            )
        ]

        colors2 = ["green" if imp > 0 else "red" for imp in relative_improvements]
        bars2 = ax2.bar(
            range(len(x)),
            relative_improvements,
            color=colors2,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        for i, (bar, imp) in enumerate(zip(bars2, relative_improvements)):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{imp:+.1f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=10,
                weight="bold",
            )

        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax2.set_xlabel("Full Label Percentage", fontsize=12, weight="bold")
        ax2.set_ylabel("Relative Improvement (%)", fontsize=12, weight="bold")
        ax2.set_title("Relative Performance Gain", fontsize=14, weight="bold")
        ax2.set_xticks(range(len(x)))
        ax2.set_xticklabels([f"{int(p)}%" for p in x])
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        return fig

    def plot_label_efficiency(self, save_name="label_efficiency.png"):
        """
        Plot label efficiency analysis showing AP per annotation effort.

        Args:
            save_name: Filename for saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        # Calculate total annotation cost (assuming full label = 1.0, point = 0.1)
        full_costs = [p / 100 for p in self.data["full_label_percent"]]
        weak_costs = [
            p / 100 * 0.1 for p in self.data["weak_label_percent"]
        ]  # Point labels cost 10% of full
        total_costs = [f + w for f, w in zip(full_costs, weak_costs)]

        # Teacher only uses full labels
        teacher_costs = full_costs

        # Plot efficiency
        ax.plot(
            teacher_costs,
            self.data["teacher_ap"],
            "o-",
            linewidth=2.5,
            markersize=8,
            label="Teacher (Full Labels Only)",
            color="#1f77b4",
        )
        ax.plot(
            total_costs,
            self.data["student_ap"],
            "s-",
            linewidth=2.5,
            markersize=8,
            label="Student (Full + Weak Labels)",
            color="#ff7f0e",
        )

        # Add annotations for key points
        for i in [0, len(total_costs) // 2, -1]:  # Start, middle, end
            ax.annotate(
                f"{self.data['proportions'][i]}",
                xy=(total_costs[i], self.data["student_ap"][i]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
            )

        ax.set_xlabel("Annotation Cost (Normalized)", fontsize=14, weight="bold")
        ax.set_ylabel("Average Precision (AP)", fontsize=14, weight="bold")
        ax.set_title(
            "Label Efficiency: Performance vs Annotation Cost",
            fontsize=16,
            weight="bold",
            pad=20,
        )

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=12, loc="lower right", framealpha=0.9)

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        return fig

    def plot_performance_heatmap(self, save_name="performance_heatmap.png"):
        """
        Plot heatmap showing all metrics across proportions.

        Args:
            save_name: Filename for saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Prepare data for heatmap
        metrics_labels = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        proportions = [f"{int(p)}%" for p in self.data["full_label_percent"]]

        # Teacher heatmap
        teacher_data = np.array(
            [
                self.data["teacher_ap"],
                self.data["teacher_ap50"],
                self.data["teacher_ap75"],
                self.data["teacher_aps"],
                self.data["teacher_apm"],
                self.data["teacher_apl"],
            ]
        )

        sns.heatmap(
            teacher_data,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            xticklabels=proportions,
            yticklabels=metrics_labels,
            cbar_kws={"label": "AP Value"},
            ax=ax1,
            vmin=0,
            vmax=70,
        )
        ax1.set_title("Teacher Network Performance", fontsize=14, weight="bold")
        ax1.set_xlabel("Full Label Percentage", fontsize=12)
        ax1.set_ylabel("Metrics", fontsize=12)

        # Student heatmap
        student_data = np.array(
            [
                self.data["student_ap"],
                self.data["student_ap50"],
                self.data["student_ap75"],
                self.data["student_aps"],
                self.data["student_apm"],
                self.data["student_apl"],
            ]
        )

        sns.heatmap(
            student_data,
            annot=True,
            fmt=".1f",
            cmap="YlGnBu",
            xticklabels=proportions,
            yticklabels=metrics_labels,
            cbar_kws={"label": "AP Value"},
            ax=ax2,
            vmin=0,
            vmax=70,
        )
        ax2.set_title(
            "Student Network Performance (w/ Weak Labels)", fontsize=14, weight="bold"
        )
        ax2.set_xlabel("Full Label Percentage", fontsize=12)
        ax2.set_ylabel("Metrics", fontsize=12)

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        return fig

    def plot_comprehensive_dashboard(self, save_name="comprehensive_dashboard.png"):
        """
        Create a comprehensive dashboard with all key visualizations.

        Args:
            save_name: Filename for saved plot
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        fig.suptitle(
            "PointWSSIS Comprehensive Performance Dashboard",
            fontsize=20,
            weight="bold",
            y=0.98,
        )

        x = self.data["full_label_percent"]

        # 1. Main AP trends (top row, span 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(
            x,
            self.data["teacher_ap"],
            "o-",
            linewidth=3,
            markersize=10,
            label="Teacher",
            color="#1f77b4",
        )
        ax1.plot(
            x,
            self.data["student_ap"],
            "s-",
            linewidth=3,
            markersize=10,
            label="Student",
            color="#ff7f0e",
        )
        ax1.set_xlabel("Full Label %", fontsize=12, weight="bold")
        ax1.set_ylabel("AP", fontsize=12, weight="bold")
        ax1.set_title("Main Performance Trend", fontsize=14, weight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{int(p)}" for p in x])

        # 2. Improvement bars (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        colors = ["green" if imp > 0 else "red" for imp in self.data["improvement"]]
        ax2.bar(range(len(x)), self.data["improvement"], color=colors, alpha=0.7)
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax2.set_xlabel("Proportion", fontsize=10, weight="bold")
        ax2.set_ylabel("AP Gain", fontsize=10, weight="bold")
        ax2.set_title("Performance Improvement", fontsize=12, weight="bold")
        ax2.set_xticks(range(len(x)))
        ax2.set_xticklabels([f"{int(p)}" for p in x], fontsize=8, rotation=45)
        ax2.grid(True, alpha=0.3, axis="y")

        # 3-5. Multi-metric trends (middle row)
        metrics_to_plot = [
            ("AP50", "teacher_ap50", "student_ap50"),
            ("AP75", "teacher_ap75", "student_ap75"),
            ("APs", "teacher_aps", "student_aps"),
        ]

        for idx, (metric_name, teacher_key, student_key) in enumerate(metrics_to_plot):
            ax = fig.add_subplot(gs[1, idx])
            ax.plot(
                x,
                self.data[teacher_key],
                "o-",
                linewidth=2,
                markersize=6,
                label="Teacher",
                alpha=0.7,
            )
            ax.plot(
                x,
                self.data[student_key],
                "s-",
                linewidth=2,
                markersize=6,
                label="Student",
                alpha=0.7,
            )
            ax.set_title(metric_name, fontsize=12, weight="bold")
            ax.set_xlabel("Full %", fontsize=9)
            ax.set_ylabel(metric_name, fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(p)}" for p in x], fontsize=7, rotation=45)

        # 6-8. Size-based AP (bottom row)
        size_metrics = [
            ("APm", "teacher_apm", "student_apm", "Medium"),
            ("APl", "teacher_apl", "student_apl", "Large"),
        ]

        for idx, (metric_name, teacher_key, student_key, size_name) in enumerate(
            size_metrics
        ):
            ax = fig.add_subplot(gs[2, idx])
            ax.plot(
                x,
                self.data[teacher_key],
                "o-",
                linewidth=2,
                markersize=6,
                label="Teacher",
                alpha=0.7,
            )
            ax.plot(
                x,
                self.data[student_key],
                "s-",
                linewidth=2,
                markersize=6,
                label="Student",
                alpha=0.7,
            )
            ax.set_title(
                f"{size_name} Objects ({metric_name})", fontsize=12, weight="bold"
            )
            ax.set_xlabel("Full %", fontsize=9)
            ax.set_ylabel(metric_name, fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(p)}" for p in x], fontsize=7, rotation=45)

        # 9. Summary table (bottom right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis("off")

        # Create summary table
        table_data = []
        for i, prop in enumerate(self.data["proportions"]):
            row = [
                f"{int(self.data['full_label_percent'][i])}%",
                f"{self.data['teacher_ap'][i]:.1f}",
                f"{self.data['student_ap'][i]:.1f}",
                f"{self.data['improvement'][i]:+.1f}",
            ]
            table_data.append(row)

        table = ax9.table(
            cellText=table_data,
            colLabels=["Full%", "Teacher", "Student", "Gain"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)

        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        ax9.set_title("Performance Summary", fontsize=12, weight="bold", pad=10)

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        return fig

    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("\n" + "=" * 80)
        print("Generating Visualizations")
        print("=" * 80 + "\n")

        plots = [
            ("AP Trends", self.plot_ap_trends),
            ("Multi-Metric Comparison", self.plot_multi_metric_comparison),
            ("Improvement Analysis", self.plot_improvement_analysis),
            ("Label Efficiency", self.plot_label_efficiency),
            ("Performance Heatmap", self.plot_performance_heatmap),
            ("Comprehensive Dashboard", self.plot_comprehensive_dashboard),
        ]

        for name, plot_func in plots:
            print(f"Generating: {name}...")
            try:
                plot_func()
                print(f"  ✓ Success\n")
            except Exception as e:
                print(f"  ✗ Error: {e}\n")

        print("=" * 80)
        print("All visualizations generated successfully!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for PointWSSIS evaluation results"
    )

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation results JSON file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as results file)",
    )

    parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "trends",
            "metrics",
            "improvement",
            "efficiency",
            "heatmap",
            "dashboard",
        ],
        help="Which plots to generate",
    )

    return parser.parse_args()


def main():
    """Main visualization pipeline."""
    args = parse_args()

    if not os.path.exists(args.results):
        print(f"Error: Results file not found: {args.results}")
        return

    print("\n" + "=" * 80)
    print("PointWSSIS Results Visualization")
    print("=" * 80)
    print(f"Results file: {args.results}")
    print(f"Output dir: {args.output_dir or 'Same as results'}")
    print("=" * 80 + "\n")

    # Create visualizer
    visualizer = ResultVisualizer(args.results, args.output_dir)

    # Generate requested plots
    if "all" in args.plots:
        visualizer.generate_all_plots()
    else:
        plot_mapping = {
            "trends": visualizer.plot_ap_trends,
            "metrics": visualizer.plot_multi_metric_comparison,
            "improvement": visualizer.plot_improvement_analysis,
            "efficiency": visualizer.plot_label_efficiency,
            "heatmap": visualizer.plot_performance_heatmap,
            "dashboard": visualizer.plot_comprehensive_dashboard,
        }

        for plot_name in args.plots:
            if plot_name in plot_mapping:
                print(f"Generating: {plot_name}")
                plot_mapping[plot_name]()

    print("\nVisualization completed successfully!")


if __name__ == "__main__":
    main()
