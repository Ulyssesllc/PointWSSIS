"""
Example: Quick Evaluation and Visualization

This script demonstrates how to use the evaluation framework with minimal setup.
"""

import os
import sys
import json
from evaluation_utils import (
    compute_metrics_statistics,
    compute_improvement_metrics,
    format_metric_table,
    export_to_csv,
    generate_latex_table,
    compare_with_baseline,
    find_best_proportion,
)


def create_sample_results():
    """
    Create sample results for demonstration.
    Replace with actual evaluation results.
    """
    sample_results = {
        "coco_5p": {
            "full_label_percent": 5,
            "weak_label_percent": 95,
            "weak_type": "P",
            "teacher_metrics": {
                "AP": 28.5,
                "AP50": 48.2,
                "AP75": 29.1,
                "APs": 12.3,
                "APm": 31.2,
                "APl": 42.1,
                "bbox_AP": 30.1,
                "bbox_AP50": 50.5,
                "bbox_AP75": 31.2,
            },
            "student_metrics": {
                "AP": 33.7,
                "AP50": 54.1,
                "AP75": 35.6,
                "APs": 16.8,
                "APm": 36.9,
                "APl": 48.3,
                "bbox_AP": 35.2,
                "bbox_AP50": 56.8,
                "bbox_AP75": 37.1,
            },
        },
        "coco_10p": {
            "full_label_percent": 10,
            "weak_label_percent": 90,
            "weak_type": "P",
            "teacher_metrics": {
                "AP": 31.2,
                "AP50": 51.5,
                "AP75": 32.8,
                "APs": 14.5,
                "APm": 34.1,
                "APl": 45.8,
                "bbox_AP": 32.8,
                "bbox_AP50": 53.2,
                "bbox_AP75": 34.5,
            },
            "student_metrics": {
                "AP": 35.8,
                "AP50": 56.9,
                "AP75": 38.2,
                "APs": 18.2,
                "APm": 39.1,
                "APl": 51.2,
                "bbox_AP": 37.5,
                "bbox_AP50": 59.1,
                "bbox_AP75": 39.8,
            },
        },
        "coco_20p": {
            "full_label_percent": 20,
            "weak_label_percent": 80,
            "weak_type": "P",
            "teacher_metrics": {
                "AP": 34.5,
                "AP50": 54.8,
                "AP75": 36.2,
                "APs": 16.8,
                "APm": 37.5,
                "APl": 49.1,
                "bbox_AP": 36.2,
                "bbox_AP50": 56.5,
                "bbox_AP75": 38.1,
            },
            "student_metrics": {
                "AP": 37.1,
                "AP50": 58.5,
                "AP75": 39.8,
                "APs": 19.5,
                "APm": 40.8,
                "APl": 52.6,
                "bbox_AP": 38.9,
                "bbox_AP50": 60.2,
                "bbox_AP75": 41.5,
            },
        },
    }

    return sample_results


def main():
    """Run example analysis."""
    print("\n" + "=" * 80)
    print("PointWSSIS Evaluation - Quick Example")
    print("=" * 80 + "\n")

    # Load or create sample results
    results_file = "evaluation_results/evaluation_results.json"

    if os.path.exists(results_file):
        print(f"Loading results from: {results_file}")
        with open(results_file, "r") as f:
            results = json.load(f)
    else:
        print("Using sample results (for demonstration)")
        results = create_sample_results()

    print(f"Found {len(results)} experiment results\n")

    # 1. Display formatted table
    print("=" * 80)
    print("1. PERFORMANCE SUMMARY TABLE")
    print("=" * 80)
    table = format_metric_table(results, metrics=["AP", "AP50", "AP75"])
    print(table)
    print()

    # 2. Compute statistics
    print("=" * 80)
    print("2. STATISTICAL SUMMARY")
    print("=" * 80)
    stats = compute_metrics_statistics(results)
    print(f"Mean Student AP:    {stats['mean'].get('AP', 0):.2f}")
    print(f"Std Student AP:     {stats['std'].get('AP', 0):.2f}")
    print(f"Min Student AP:     {stats['min'].get('AP', 0):.2f}")
    print(f"Max Student AP:     {stats['max'].get('AP', 0):.2f}")
    print(f"Median Student AP:  {stats['median'].get('AP', 0):.2f}")
    print()

    # 3. Improvement analysis
    print("=" * 80)
    print("3. IMPROVEMENT ANALYSIS")
    print("=" * 80)
    for proportion, data in results.items():
        teacher = data.get("teacher_metrics", {})
        student = data.get("student_metrics", {})
        improvements = compute_improvement_metrics(teacher, student)

        print(f"\n{proportion}:")
        print(f"  Teacher AP: {teacher.get('AP', 0):.2f}")
        print(f"  Student AP: {student.get('AP', 0):.2f}")
        print(
            f"  Absolute improvement: {improvements.get('AP_abs_improvement', 0):+.2f}"
        )
        print(
            f"  Relative improvement: {improvements.get('AP_rel_improvement', 0):+.2f}%"
        )
    print()

    # 4. Compare with baseline
    print("=" * 80)
    print("4. BASELINE COMPARISON (Fully Supervised = 39.7 AP)")
    print("=" * 80)
    comparisons = compare_with_baseline(results, baseline_ap=39.7)

    for proportion, comp in sorted(
        comparisons.items(), key=lambda x: results[x[0]]["full_label_percent"]
    ):
        print(f"{proportion}:")
        print(f"  Student AP: {comp['student_ap']:.2f}")
        print(f"  % of baseline: {comp['pct_of_baseline']:.1f}%")
        print(f"  Gap: {comp['gap']:.2f}")
        print(f"  Annotation cost: {comp['annotation_cost']:.3f}")
        print(f"  Efficiency: {comp['efficiency']:.2f}")
        print()

    # 5. Find best configuration
    print("=" * 80)
    print("5. BEST CONFIGURATIONS")
    print("=" * 80)

    # Best performance
    best_perf, best_perf_data = find_best_proportion(
        results, metric="AP", criterion="performance"
    )
    print(f"Best Performance: {best_perf}")
    print(f"  AP: {best_perf_data['student_metrics']['AP']:.2f}")

    # Best improvement
    best_imp, best_imp_data = find_best_proportion(
        results, metric="AP", criterion="improvement"
    )
    teacher_ap = best_imp_data["teacher_metrics"]["AP"]
    student_ap = best_imp_data["student_metrics"]["AP"]
    print(f"\nBest Improvement: {best_imp}")
    print(f"  Gain: {student_ap - teacher_ap:+.2f}")

    # Best efficiency
    best_eff, best_eff_data = find_best_proportion(
        results, metric="AP", criterion="efficiency"
    )
    print(f"\nBest Efficiency: {best_eff}")
    print(f"  AP: {best_eff_data['student_metrics']['AP']:.2f}")
    print()

    # 6. Export results
    print("=" * 80)
    print("6. EXPORTING RESULTS")
    print("=" * 80)

    # Export to CSV
    csv_path = "evaluation_results/results_export.csv"
    os.makedirs("evaluation_results", exist_ok=True)
    export_to_csv(results, csv_path)
    print(f"✓ Exported to CSV: {csv_path}")

    # Generate LaTeX table
    latex_table = generate_latex_table(results, metrics=["AP", "AP50", "AP75"])
    latex_path = "evaluation_results/results_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"✓ Generated LaTeX table: {latex_path}")

    # Save JSON
    json_path = "evaluation_results/results_analysis.json"
    analysis_results = {
        "statistics": stats,
        "baseline_comparison": comparisons,
        "best_performance": best_perf,
        "best_improvement": best_imp,
        "best_efficiency": best_eff,
    }
    with open(json_path, "w") as f:
        json.dump(analysis_results, f, indent=4)
    print(f"✓ Saved analysis: {json_path}")
    print()

    # 7. Next steps
    print("=" * 80)
    print("7. NEXT STEPS")
    print("=" * 80)
    print("To generate visualizations, run:")
    print(f"  python visualize_results.py --results {results_file}")
    print("\nTo run full evaluation pipeline:")
    print("  bash run_evaluation.sh --data_root /path/to/coco")
    print("\nFor more details, see:")
    print("  cat EVALUATION_GUIDE.md")
    print()

    print("=" * 80)
    print("Example completed successfully! ✓")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
