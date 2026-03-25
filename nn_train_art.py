"""Train and evaluate ART variants with one shared workflow."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from nn_model_art import (
    ALPHABET_A_TO_T,
    MODEL_TYPES,
    BaseARTCharacterModel,
    ModelType,
    create_initial_model,
    default_model_path,
    discover_pattern_images,
    load_pattern_vector,
)


@dataclass
class TrainingReport:
    model_type: ModelType
    epochs: int
    augment_per_symbol: int
    samples_seen: int
    clean_accuracy: float
    noisy_accuracy: float
    model_path: Path


def _flip_bits(vector: Sequence[float], flips: int, rng: random.Random) -> List[float]:
    augmented = list(vector)
    flips = max(0, min(flips, len(augmented)))
    for index in rng.sample(range(len(augmented)), flips):
        augmented[index] = 0.0 if augmented[index] > 0.5 else 1.0
    return augmented


def _evaluate(model: BaseARTCharacterModel, samples: Dict[str, List[List[float]]]) -> float:
    total = 0
    correct = 0
    for label, vectors in samples.items():
        for vector in vectors:
            prediction = model.predict(vector)
            total += 1
            if prediction.label == label:
                correct += 1
    if total == 0:
        return 0.0
    return correct / total


def _build_training_samples(pattern_dir: Path) -> Dict[str, List[float]]:
    image_map = discover_pattern_images(pattern_dir)
    missing = [label for label in ALPHABET_A_TO_T if label not in image_map]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing pattern files for labels: {missing_text}. Expected files in '{pattern_dir}'."
        )

    return {label: load_pattern_vector(image_map[label]) for label in ALPHABET_A_TO_T}


def train_model(
    model_type: ModelType,
    pattern_dir: Path,
    model_path: Path,
    epochs: int = 20,
    augment_per_symbol: int = 10,
    flips_per_sample: int = 2,
    seed: int = 498,
    vigilance: float = 0.82,
    learning_rate: float = 0.6,
    choice_alpha: float = 1e-3,
) -> TrainingReport:
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if augment_per_symbol < 0:
        raise ValueError("augment_per_symbol must be >= 0")

    rng = random.Random(seed)
    base_vectors = _build_training_samples(pattern_dir)

    model = create_initial_model(
        model_type=model_type,
        pattern_dir=pattern_dir,
        vigilance=vigilance,
        learning_rate=learning_rate,
        choice_alpha=choice_alpha,
    )

    samples_seen = 0
    use_augmentation = model_type == "aug_fuzzy_art"

    for _ in range(epochs):
        labels = list(ALPHABET_A_TO_T)
        rng.shuffle(labels)
        for label in labels:
            base_vector = base_vectors[label]
            model.train_pattern(base_vector, label, augmented=False)
            samples_seen += 1

            if use_augmentation:
                for _ in range(augment_per_symbol):
                    augmented = _flip_bits(base_vector, flips_per_sample, rng)
                    model.train_pattern(augmented, label, augmented=True)
                    samples_seen += 1

    clean_eval = {label: [vector] for label, vector in base_vectors.items()}

    noisy_eval: Dict[str, List[List[float]]] = {}
    for label, base_vector in base_vectors.items():
        noisy_eval[label] = [_flip_bits(base_vector, flips_per_sample, rng) for _ in range(10)]

    clean_accuracy = _evaluate(model, clean_eval)
    noisy_accuracy = _evaluate(model, noisy_eval)

    model.set_training_metrics(
        clean_accuracy=clean_accuracy,
        noisy_accuracy=noisy_accuracy,
        samples_seen=samples_seen,
    )
    model.save(model_path)

    return TrainingReport(
        model_type=model_type,
        epochs=epochs,
        augment_per_symbol=augment_per_symbol if use_augmentation else 0,
        samples_seen=samples_seen,
        clean_accuracy=clean_accuracy,
        noisy_accuracy=noisy_accuracy,
        model_path=model_path,
    )


def train_models(
    model_types: Sequence[ModelType],
    pattern_dir: Path,
    model_dir: Path,
    epochs: int = 20,
    augment_per_symbol: int = 10,
    flips_per_sample: int = 2,
    seed: int = 498,
    vigilance: float = 0.82,
    learning_rate: float = 0.6,
    choice_alpha: float = 1e-3,
) -> Dict[ModelType, TrainingReport]:
    reports: Dict[ModelType, TrainingReport] = {}

    for model_type in model_types:
        path = model_dir / default_model_path(model_type).name
        report = train_model(
            model_type=model_type,
            pattern_dir=pattern_dir,
            model_path=path,
            epochs=epochs,
            augment_per_symbol=augment_per_symbol,
            flips_per_sample=flips_per_sample,
            seed=seed,
            vigilance=vigilance,
            learning_rate=learning_rate,
            choice_alpha=choice_alpha,
        )
        reports[model_type] = report

    return reports


def _format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"


def _print_report(report: TrainingReport) -> None:
    print(f"Model type: {report.model_type}")
    print(f"Model saved to: {report.model_path}")
    print(f"Epochs: {report.epochs}")
    print(f"Augment/sample per symbol: {report.augment_per_symbol}")
    print(f"Samples seen: {report.samples_seen}")
    print(f"Clean accuracy: {_format_accuracy(report.clean_accuracy)}")
    print(f"Noisy accuracy: {_format_accuracy(report.noisy_accuracy)}")


def _print_comparison_table(reports: Dict[ModelType, TrainingReport]) -> None:
    print("\nComparison (sorted by noisy accuracy)")
    print("Model           Clean      Noisy      Samples")
    print("------------------------------------------------")

    sorted_reports = sorted(reports.values(), key=lambda item: item.noisy_accuracy, reverse=True)
    for report in sorted_reports:
        print(
            f"{report.model_type:<14}"
            f"{_format_accuracy(report.clean_accuracy):<11}"
            f"{_format_accuracy(report.noisy_accuracy):<11}"
            f"{report.samples_seen}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ART models for A-T pattern recognition")
    parser.add_argument("--pattern-dir", type=Path, default=Path("patterns_orig"))
    parser.add_argument("--model-dir", type=Path, default=Path("patterns_Ass4"))
    parser.add_argument("--model-type", type=str, default="all", choices=["all", *MODEL_TYPES])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--augment-per-symbol", type=int, default=10)
    parser.add_argument("--flips-per-sample", type=int, default=2)
    parser.add_argument("--seed", type=int, default=498)
    parser.add_argument("--vigilance", type=float, default=0.82)
    parser.add_argument("--learning-rate", type=float, default=0.6)
    parser.add_argument("--choice-alpha", type=float, default=1e-3)

    args = parser.parse_args()

    if args.model_type == "all":
        selected_types: Sequence[ModelType] = MODEL_TYPES
    else:
        selected_types = [args.model_type]

    reports = train_models(
        model_types=selected_types,
        pattern_dir=args.pattern_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        augment_per_symbol=args.augment_per_symbol,
        flips_per_sample=args.flips_per_sample,
        seed=args.seed,
        vigilance=args.vigilance,
        learning_rate=args.learning_rate,
        choice_alpha=args.choice_alpha,
    )

    print("Training complete")
    if len(reports) == 1:
        only = next(iter(reports.values()))
        _print_report(only)
    else:
        for model_type in MODEL_TYPES:
            if model_type in reports:
                print("\n---")
                _print_report(reports[model_type])
        _print_comparison_table(reports)


if __name__ == "__main__":
    main()
