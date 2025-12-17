"""
Utility Functions for PointWSSIS Evaluation

This module provides helper functions for:
- Loading and processing evaluation results
- Computing metrics and statistics
- Data formatting and conversion
- Result aggregation and comparison
"""

import os
import json
import yaml
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import warnings


def load_yaml_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_json_results(results_path: str) -> Dict:
    """
    Load JSON results file.

    Args:
        results_path: Path to JSON results file

    Returns:
        Dictionary containing results
    """
    with open(results_path, "r") as f:
        results = json.load(f)
    return results


def save_json(data: Dict, save_path: str, indent: int = 4):
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        save_path: Output file path
        indent: JSON indentation level
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=indent)


def compute_metrics_statistics(results: Dict[str, Dict]) -> Dict:
    """
    Compute statistics across multiple evaluation results.

    Args:
        results: Dictionary of results keyed by proportion name

    Returns:
        Dictionary containing statistical summaries
    """
    stats = {"mean": {}, "std": {}, "min": {}, "max": {}, "median": {}}

    # Collect all metric values
    metric_values = defaultdict(list)

    for proportion, data in results.items():
        if "student_metrics" in data:
            for metric, value in data["student_metrics"].items():
                if isinstance(value, (int, float)):
                    metric_values[metric].append(value)

    # Compute statistics for each metric
    for metric, values in metric_values.items():
        if values:
            stats["mean"][metric] = np.mean(values)
            stats["std"][metric] = np.std(values)
            stats["min"][metric] = np.min(values)
            stats["max"][metric] = np.max(values)
            stats["median"][metric] = np.median(values)

    return stats


def compute_improvement_metrics(teacher_metrics: Dict, student_metrics: Dict) -> Dict:
    """
    Compute improvement metrics between teacher and student models.

    Args:
        teacher_metrics: Teacher model metrics
        student_metrics: Student model metrics

    Returns:
        Dictionary containing improvement metrics
    """
    improvements = {}

    for metric in teacher_metrics.keys():
        if metric in student_metrics:
            teacher_val = teacher_metrics[metric]
            student_val = student_metrics[metric]

            # Absolute improvement
            abs_improvement = student_val - teacher_val
            improvements[f"{metric}_abs_improvement"] = abs_improvement

            # Relative improvement (percentage)
            if teacher_val > 0:
                rel_improvement = (abs_improvement / teacher_val) * 100
                improvements[f"{metric}_rel_improvement"] = rel_improvement
            else:
                improvements[f"{metric}_rel_improvement"] = 0.0

    return improvements


def compute_annotation_cost(
    full_label_percent: float,
    weak_label_percent: float,
    weak_type: str = "P",
    cost_model: Optional[Dict] = None,
) -> float:
    """
    Compute normalized annotation cost.

    Args:
        full_label_percent: Percentage of full labels (0-100)
        weak_label_percent: Percentage of weak labels (0-100)
        weak_type: Type of weak label ('U', 'I', 'P', 'B', 'F')
        cost_model: Dictionary mapping label types to costs

    Returns:
        Normalized annotation cost
    """
    if cost_model is None:
        cost_model = {
            "F": 1.0,  # Full mask
            "B": 0.3,  # Box
            "P": 0.1,  # Point
            "I": 0.05,  # Image-level
            "U": 0.0,  # Unlabeled
        }

    full_cost = (full_label_percent / 100) * cost_model["F"]
    weak_cost = (weak_label_percent / 100) * cost_model.get(weak_type, 0.0)

    total_cost = full_cost + weak_cost

    return total_cost


def compute_efficiency_score(
    ap_value: float, annotation_cost: float, baseline_ap: float = 39.7
) -> float:
    """
    Compute efficiency score (performance per unit cost).

    Args:
        ap_value: AP value achieved
        annotation_cost: Normalized annotation cost
        baseline_ap: Baseline fully supervised AP

    Returns:
        Efficiency score
    """
    if annotation_cost == 0:
        return 0.0

    # Normalize AP relative to baseline
    normalized_ap = ap_value / baseline_ap

    # Efficiency = performance / cost
    efficiency = normalized_ap / annotation_cost

    return efficiency


def format_metric_table(results: Dict[str, Dict], metrics: List[str] = None) -> str:
    """
    Format results as a text table.

    Args:
        results: Dictionary of results
        metrics: List of metrics to include (default: ['AP', 'AP50', 'AP75'])

    Returns:
        Formatted table string
    """
    if metrics is None:
        metrics = ["AP", "AP50", "AP75"]

    # Header
    header = f"{'Proportion':<15} | "
    for metric in metrics:
        header += f"{'T_' + metric:<8} | {'S_' + metric:<8} | {'Δ_' + metric:<8} | "

    separator = "-" * len(header)

    table = separator + "\n" + header + "\n" + separator + "\n"

    # Rows
    sorted_results = sorted(
        results.items(), key=lambda x: x[1].get("full_label_percent", 0)
    )

    for proportion, data in sorted_results:
        teacher = data.get("teacher_metrics", {})
        student = data.get("student_metrics", {})

        row = f"{proportion:<15} | "

        for metric in metrics:
            t_val = teacher.get(metric, 0.0)
            s_val = student.get(metric, 0.0)
            delta = s_val - t_val

            row += f"{t_val:<8.2f} | {s_val:<8.2f} | {delta:+<8.2f} | "

        table += row + "\n"

    table += separator

    return table


def compare_with_baseline(
    results: Dict[str, Dict],
    baseline_ap: float = 39.7,
    baseline_name: str = "Fully Supervised",
) -> Dict:
    """
    Compare results with a fully supervised baseline.

    Args:
        results: Dictionary of evaluation results
        baseline_ap: Baseline AP value
        baseline_name: Name of baseline

    Returns:
        Dictionary containing comparison metrics
    """
    comparisons = {}

    for proportion, data in results.items():
        student_ap = data.get("student_metrics", {}).get("AP", 0.0)

        # Absolute gap
        gap = baseline_ap - student_ap

        # Percentage of baseline achieved
        pct_of_baseline = (student_ap / baseline_ap) * 100 if baseline_ap > 0 else 0

        # Annotation cost
        cost = compute_annotation_cost(
            data.get("full_label_percent", 0),
            data.get("weak_label_percent", 0),
            data.get("weak_type", "P"),
        )

        comparisons[proportion] = {
            "student_ap": student_ap,
            "baseline_ap": baseline_ap,
            "gap": gap,
            "pct_of_baseline": pct_of_baseline,
            "annotation_cost": cost,
            "efficiency": compute_efficiency_score(student_ap, cost, baseline_ap),
        }

    return comparisons


def aggregate_results_by_weak_type(results: Dict[str, Dict]) -> Dict[str, List]:
    """
    Aggregate results by weak label type.

    Args:
        results: Dictionary of evaluation results

    Returns:
        Dictionary mapping weak types to list of results
    """
    aggregated = defaultdict(list)

    for proportion, data in results.items():
        weak_type = data.get("weak_type", "P")
        aggregated[weak_type].append(
            {
                "proportion": proportion,
                "full_percent": data.get("full_label_percent", 0),
                "weak_percent": data.get("weak_label_percent", 0),
                "teacher_metrics": data.get("teacher_metrics", {}),
                "student_metrics": data.get("student_metrics", {}),
            }
        )

    return dict(aggregated)


def compute_area_under_curve(x: List[float], y: List[float]) -> float:
    """
    Compute area under curve using trapezoidal rule.

    Args:
        x: X-axis values (sorted)
        y: Y-axis values

    Returns:
        Area under curve
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    # Sort by x
    sorted_pairs = sorted(zip(x, y))
    x_sorted = [p[0] for p in sorted_pairs]
    y_sorted = [p[1] for p in sorted_pairs]

    # Trapezoidal rule
    auc = np.trapz(y_sorted, x_sorted)

    return auc


def find_best_proportion(
    results: Dict[str, Dict], metric: str = "AP", criterion: str = "efficiency"
) -> Tuple[str, Dict]:
    """
    Find the best proportion based on a criterion.

    Args:
        results: Dictionary of evaluation results
        metric: Metric to use (default: 'AP')
        criterion: Selection criterion ('efficiency', 'performance', 'improvement')

    Returns:
        Tuple of (proportion_name, proportion_data)
    """
    best_proportion = None
    best_score = -float("inf")

    for proportion, data in results.items():
        if criterion == "performance":
            score = data.get("student_metrics", {}).get(metric, 0.0)

        elif criterion == "improvement":
            teacher_val = data.get("teacher_metrics", {}).get(metric, 0.0)
            student_val = data.get("student_metrics", {}).get(metric, 0.0)
            score = student_val - teacher_val

        elif criterion == "efficiency":
            student_ap = data.get("student_metrics", {}).get(metric, 0.0)
            cost = compute_annotation_cost(
                data.get("full_label_percent", 0),
                data.get("weak_label_percent", 0),
                data.get("weak_type", "P"),
            )
            score = compute_efficiency_score(student_ap, cost)

        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if score > best_score:
            best_score = score
            best_proportion = proportion

    return best_proportion, results[best_proportion]


def generate_latex_table(results: Dict[str, Dict], metrics: List[str] = None) -> str:
    """
    Generate LaTeX table from results.

    Args:
        results: Dictionary of evaluation results
        metrics: List of metrics to include

    Returns:
        LaTeX table string
    """
    if metrics is None:
        metrics = ["AP", "AP50", "AP75"]

    # Start table
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{PointWSSIS Evaluation Results}\n"
    latex += "\\label{tab:pointwssis_results}\n"

    # Table format
    num_cols = (
        2 + len(metrics) * 2
    )  # proportion + full% + (teacher + student) * metrics
    latex += "\\begin{tabular}{l" + "c" * (num_cols - 1) + "}\n"
    latex += "\\toprule\n"

    # Header
    latex += "Proportion & Full\\% "
    for metric in metrics:
        latex += f"& Teacher & Student "
    latex += "\\\\\n"

    # Subheader with metric names
    latex += "& "
    for metric in metrics:
        latex += f"& \\multicolumn{{2}}{{c}}{{{metric}}} "
    latex += "\\\\\n"
    latex += "\\midrule\n"

    # Rows
    sorted_results = sorted(
        results.items(), key=lambda x: x[1].get("full_label_percent", 0)
    )

    for proportion, data in sorted_results:
        full_pct = data.get("full_label_percent", 0)
        teacher = data.get("teacher_metrics", {})
        student = data.get("student_metrics", {})

        latex += f"{proportion} & {full_pct}\\% "

        for metric in metrics:
            t_val = teacher.get(metric, 0.0)
            s_val = student.get(metric, 0.0)

            # Bold if student > teacher
            if s_val > t_val:
                latex += f"& {t_val:.1f} & \\textbf{{{s_val:.1f}}} "
            else:
                latex += f"& {t_val:.1f} & {s_val:.1f} "

        latex += "\\\\\n"

    # End table
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def validate_results(results: Dict[str, Dict]) -> Tuple[bool, List[str]]:
    """
    Validate evaluation results for completeness and consistency.

    Args:
        results: Dictionary of evaluation results

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings_list = []
    is_valid = True

    required_keys = ["full_label_percent", "weak_label_percent", "weak_type"]

    for proportion, data in results.items():
        # Check required keys
        for key in required_keys:
            if key not in data:
                warnings_list.append(f"{proportion}: Missing key '{key}'")
                is_valid = False

        # Check metrics
        if "student_metrics" not in data and "teacher_metrics" not in data:
            warnings_list.append(f"{proportion}: No metrics found")
            is_valid = False

        # Check percentage sum
        full_pct = data.get("full_label_percent", 0)
        weak_pct = data.get("weak_label_percent", 0)
        if abs((full_pct + weak_pct) - 100) > 1e-6:
            warnings_list.append(
                f"{proportion}: Percentages don't sum to 100 ({full_pct} + {weak_pct})"
            )

        # Check metric ranges
        for metrics_key in ["teacher_metrics", "student_metrics"]:
            if metrics_key in data:
                for metric, value in data[metrics_key].items():
                    if "AP" in metric and (value < 0 or value > 100):
                        warnings_list.append(
                            f"{proportion}: {metrics_key}.{metric} out of range: {value}"
                        )

    return is_valid, warnings_list


def export_to_csv(results: Dict[str, Dict], output_path: str):
    """
    Export results to CSV file.

    Args:
        results: Dictionary of evaluation results
        output_path: Path to output CSV file
    """
    import csv

    # Determine all metrics
    all_metrics = set()
    for data in results.values():
        if "teacher_metrics" in data:
            all_metrics.update(data["teacher_metrics"].keys())
        if "student_metrics" in data:
            all_metrics.update(data["student_metrics"].keys())

    all_metrics = sorted(all_metrics)

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["Proportion", "Full%", "Weak%", "Type"]
        for metric in all_metrics:
            header.extend([f"Teacher_{metric}", f"Student_{metric}", f"Δ_{metric}"])
        writer.writerow(header)

        # Rows
        sorted_results = sorted(
            results.items(), key=lambda x: x[1].get("full_label_percent", 0)
        )

        for proportion, data in sorted_results:
            row = [
                proportion,
                data.get("full_label_percent", ""),
                data.get("weak_label_percent", ""),
                data.get("weak_type", ""),
            ]

            teacher = data.get("teacher_metrics", {})
            student = data.get("student_metrics", {})

            for metric in all_metrics:
                t_val = teacher.get(metric, "")
                s_val = student.get(metric, "")
                delta = s_val - t_val if (t_val and s_val) else ""

                row.extend([t_val, s_val, delta])

            writer.writerow(row)

    print(f"Results exported to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("PointWSSIS Evaluation Utilities")
    print("This module provides utility functions for evaluation.")
    print("\nExample functions:")
    print("  - load_yaml_config()")
    print("  - compute_metrics_statistics()")
    print("  - format_metric_table()")
    print("  - generate_latex_table()")
    print("  - export_to_csv()")
