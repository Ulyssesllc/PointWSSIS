"""
PointWSSIS Evaluation Pipeline
Evaluates instance segmentation performance across different weak label proportions.

This script evaluates models trained with different weak supervision configurations:
- U: Unlabeled data
- I: Image-level labels
- P: Point labels
- B: Box labels
- F: Full mask labels

Metrics computed: AP, AP50, AP75, APs, APm, APl
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings(action="ignore")

import torch
import numpy as np
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# Add detectron2 and AdelaiDet to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "detectron2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AdelaiDet"))

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger

from adet.config import get_cfg as get_adet_cfg


class WeakLabelEvaluator:
    """
    Evaluator for different weak supervision scenarios in instance segmentation.
    """

    # Label type definitions
    LABEL_TYPES = {
        "U": "Unlabeled",
        "I": "Image-level",
        "P": "Point",
        "B": "Box",
        "F": "Full mask",
    }

    def __init__(self, data_root, output_dir, config_file=None):
        """
        Args:
            data_root: Root directory containing COCO dataset
            output_dir: Directory to save evaluation results
            config_file: Path to model config file (optional)
        """
        self.data_root = data_root
        self.output_dir = output_dir
        self.config_file = config_file

        os.makedirs(output_dir, exist_ok=True)
        setup_logger(output=os.path.join(output_dir, "evaluation_log.txt"))

        # Store results
        self.results = defaultdict(dict)

    def setup_cfg(self, config_file, model_weights, prompt_type="point"):
        """
        Setup configuration for evaluation.

        Args:
            config_file: Path to config yaml file
            model_weights: Path to model checkpoint
            prompt_type: Type of prompt (point, box, etc.)
        """
        cfg = get_adet_cfg()
        cfg.merge_from_file(config_file)

        cfg.MODEL.WEIGHTS = model_weights
        cfg.MODEL.SOLOV2.PROMPT = prompt_type
        cfg.MODEL.SOLOV2.NMS_TYPE = "mask"
        cfg.MODEL.SOLOV2.FPN_SCALE_RANGES = (
            (1, 100000),
            (1, 100000),
            (1, 100000),
            (1, 100000),
            (1, 100000),
        )

        cfg.DATASETS.TEST = ("coco_2017_val",)
        cfg.DATALOADER.NUM_WORKERS = 4
        cfg.SOLVER.IMS_PER_BATCH = 1

        cfg.freeze()
        return cfg

    def evaluate_model(
        self,
        model_name,
        config_file,
        model_weights,
        dataset_name="coco_2017_val",
        prompt_type="point",
    ):
        """
        Evaluate a single model on the test dataset.

        Args:
            model_name: Name identifier for the model
            config_file: Path to config file
            model_weights: Path to model checkpoint
            dataset_name: Name of test dataset
            prompt_type: Type of supervision prompt

        Returns:
            Dictionary containing AP metrics
        """
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model_name}")
        print(f"Config: {config_file}")
        print(f"Weights: {model_weights}")
        print(f"{'=' * 80}\n")

        # Setup configuration
        cfg = self.setup_cfg(config_file, model_weights, prompt_type)

        # Build predictor and evaluator
        predictor = DefaultPredictor(cfg)

        # Build test loader
        test_loader = build_detection_test_loader(cfg, dataset_name)

        # Setup COCO evaluator
        evaluator = COCOEvaluator(
            dataset_name,
            cfg,
            False,
            output_dir=os.path.join(self.output_dir, model_name),
        )

        # Run evaluation
        results = inference_on_dataset(predictor.model, test_loader, evaluator)

        # Extract key metrics
        metrics = self._extract_metrics(results)

        print(f"\nResults for {model_name}:")
        self._print_metrics(metrics)

        return metrics

    def _extract_metrics(self, coco_eval_results):
        """
        Extract and format metrics from COCO evaluation results.

        Args:
            coco_eval_results: Results dictionary from COCOEvaluator

        Returns:
            Dictionary with formatted metrics
        """
        if "bbox" in coco_eval_results:
            bbox_results = coco_eval_results["bbox"]
        else:
            bbox_results = {}

        if "segm" in coco_eval_results:
            segm_results = coco_eval_results["segm"]
        else:
            segm_results = {}

        metrics = {
            "AP": segm_results.get("AP", 0.0),
            "AP50": segm_results.get("AP50", 0.0),
            "AP75": segm_results.get("AP75", 0.0),
            "APs": segm_results.get("APs", 0.0),
            "APm": segm_results.get("APm", 0.0),
            "APl": segm_results.get("APl", 0.0),
            "bbox_AP": bbox_results.get("AP", 0.0),
            "bbox_AP50": bbox_results.get("AP50", 0.0),
            "bbox_AP75": bbox_results.get("AP75", 0.0),
        }

        return metrics

    def _print_metrics(self, metrics):
        """Pretty print metrics."""
        print(f"  Segmentation Metrics:")
        print(f"    AP      : {metrics['AP']:.2f}")
        print(f"    AP50    : {metrics['AP50']:.2f}")
        print(f"    AP75    : {metrics['AP75']:.2f}")
        print(f"    APs     : {metrics['APs']:.2f}")
        print(f"    APm     : {metrics['APm']:.2f}")
        print(f"    APl     : {metrics['APl']:.2f}")
        print(f"  Detection Metrics:")
        print(f"    bbox_AP : {metrics['bbox_AP']:.2f}")
        print(f"    bbox_AP50: {metrics['bbox_AP50']:.2f}")
        print(f"    bbox_AP75: {metrics['bbox_AP75']:.2f}")

    def evaluate_proportions(self, experiment_config):
        """
        Evaluate models trained with different proportions of weak labels.

        Args:
            experiment_config: Dictionary containing experiment configurations
            Format:
            {
                'proportion_name': {
                    'teacher_weights': path,
                    'student_weights': path,
                    'config_file': path,
                    'full_label_percent': int,
                    'weak_label_percent': int,
                    'weak_type': 'P' or 'B' or 'I'
                }
            }
        """
        all_results = {}

        for proportion, config in experiment_config.items():
            print(f"\n{'#' * 80}")
            print(f"# Evaluating Proportion: {proportion}")
            print(f"# Full labels: {config['full_label_percent']}%")
            print(
                f"# Weak labels: {config['weak_label_percent']}% ({self.LABEL_TYPES[config['weak_type']]})"
            )
            print(f"{'#' * 80}\n")

            results = {}

            # Evaluate teacher network
            if "teacher_weights" in config and os.path.exists(
                config["teacher_weights"]
            ):
                teacher_metrics = self.evaluate_model(
                    model_name=f"{proportion}_teacher",
                    config_file=config["config_file"],
                    model_weights=config["teacher_weights"],
                    prompt_type="point",
                )
                results["teacher"] = teacher_metrics
            else:
                print(f"Warning: Teacher weights not found for {proportion}")

            # Evaluate student network
            if "student_weights" in config and os.path.exists(
                config["student_weights"]
            ):
                student_metrics = self.evaluate_model(
                    model_name=f"{proportion}_student",
                    config_file=config.get(
                        "student_config_file", config["config_file"]
                    ),
                    model_weights=config["student_weights"],
                    prompt_type="point",
                )
                results["student"] = student_metrics
            else:
                print(f"Warning: Student weights not found for {proportion}")

            # Store results with metadata
            all_results[proportion] = {
                "results": results,
                "config": config,
                "full_label_percent": config["full_label_percent"],
                "weak_label_percent": config["weak_label_percent"],
                "weak_type": config["weak_type"],
            }

        self.results = all_results
        return all_results

    def save_results(self, filename="evaluation_results.json"):
        """
        Save evaluation results to JSON file.

        Args:
            filename: Name of output file
        """
        output_path = os.path.join(self.output_dir, filename)

        # Convert results to serializable format
        serializable_results = {}
        for proportion, data in self.results.items():
            serializable_results[proportion] = {
                "full_label_percent": data["full_label_percent"],
                "weak_label_percent": data["weak_label_percent"],
                "weak_type": data["weak_type"],
                "teacher_metrics": data["results"].get("teacher", {}),
                "student_metrics": data["results"].get("student", {}),
            }

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=4)

        print(f"\n{'=' * 80}")
        print(f"Results saved to: {output_path}")
        print(f"{'=' * 80}\n")

        return output_path

    def generate_summary_table(self):
        """
        Generate a summary table of all evaluation results.

        Returns:
            Formatted string table
        """
        if not self.results:
            return "No results to summarize."

        # Header
        table = "\n" + "=" * 120 + "\n"
        table += f"{'Proportion':<15} {'Full%':<8} {'Weak%':<8} {'Type':<6} "
        table += f"{'Teacher AP':<12} {'Student AP':<12} {'Improvement':<12}\n"
        table += "=" * 120 + "\n"

        # Sort by full label percentage
        sorted_results = sorted(
            self.results.items(), key=lambda x: x[1]["full_label_percent"]
        )

        for proportion, data in sorted_results:
            full_pct = data["full_label_percent"]
            weak_pct = data["weak_label_percent"]
            weak_type = data["weak_type"]

            teacher_ap = data["results"].get("teacher", {}).get("AP", 0.0)
            student_ap = data["results"].get("student", {}).get("AP", 0.0)
            improvement = student_ap - teacher_ap

            table += f"{proportion:<15} {full_pct:<8} {weak_pct:<8} {weak_type:<6} "
            table += f"{teacher_ap:<12.2f} {student_ap:<12.2f} {improvement:+<12.2f}\n"

        table += "=" * 120 + "\n"

        return table


def load_experiment_config(config_path):
    """
    Load experiment configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary of experiment configurations
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_default_config(data_root, training_dir):
    """
    Create a default experiment configuration.

    Args:
        data_root: Root directory of dataset
        training_dir: Directory containing trained models

    Returns:
        Dictionary of default experiment configurations
    """
    config_file = "AdelaiDet/configs/PointWSSIS/R101_teacher.yaml"

    experiments = {}

    # COCO proportions: 1%, 2%, 5%, 10%, 20%, 30%, 50%
    proportions = [(1, 99), (2, 98), (5, 95), (10, 90), (20, 80), (30, 70), (50, 50)]

    for full_pct, weak_pct in proportions:
        exp_name = f"coco_{full_pct}p"

        experiments[exp_name] = {
            "config_file": config_file,
            "teacher_weights": f"{training_dir}/SOLOv2_R101_coco{full_pct}p_teacher/model_final.pth",
            "student_weights": f"{training_dir}/SOLOv2_R101_coco{full_pct}p_student/model_final.pth",
            "full_label_percent": full_pct,
            "weak_label_percent": weak_pct,
            "weak_type": "P",  # Point labels
        }

    return experiments


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate PointWSSIS pipeline across different weak label proportions"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to COCO dataset root directory",
    )

    parser.add_argument(
        "--training_dir",
        type=str,
        default="training_dir",
        help="Directory containing trained model checkpoints",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment configuration JSON file (optional)",
    )

    parser.add_argument(
        "--proportions",
        type=str,
        nargs="+",
        default=None,
        help="Specific proportions to evaluate (e.g., 5p 10p 20p)",
    )

    return parser.parse_args()


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    # Set environment variable for dataset path
    os.environ["DETECTRON2_DATASETS"] = args.data_root

    print("\n" + "=" * 80)
    print("PointWSSIS Evaluation Pipeline")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Training dir: {args.training_dir}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80 + "\n")

    # Initialize evaluator
    evaluator = WeakLabelEvaluator(data_root=args.data_root, output_dir=args.output_dir)

    # Load or create experiment configuration
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from: {args.config}")
        experiment_config = load_experiment_config(args.config)
    else:
        print("Using default configuration")
        experiment_config = create_default_config(args.data_root, args.training_dir)

    # Filter by specific proportions if requested
    if args.proportions:
        experiment_config = {
            k: v
            for k, v in experiment_config.items()
            if any(p in k for p in args.proportions)
        }
        print(f"Evaluating proportions: {list(experiment_config.keys())}")

    # Run evaluation
    results = evaluator.evaluate_proportions(experiment_config)

    # Save results
    results_path = evaluator.save_results()

    # Print summary
    summary = evaluator.generate_summary_table()
    print(summary)

    # Save summary to text file
    summary_path = os.path.join(args.output_dir, "summary_table.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary table saved to: {summary_path}")

    print("\nEvaluation completed successfully!")

    return results


if __name__ == "__main__":
    main()
