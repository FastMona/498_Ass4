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
    normalize_model_type,
)


DISPLAY_ORDER: Sequence[ModelType] = (
    "ART_sing",
    "ART_1",
    "fuzzy_ART",
    "aug_fuz_ART",
)

DISPLAY_NAME: Dict[ModelType, str] = {
    "ART_sing": "ART_sing",
    "ART_1": "ART_1",
    "fuzzy_ART": "fuzzy_ART",
    "aug_fuz_ART": "aug_fuz_ART",
}

DEFAULT_AUGMENTED_TRAINING_FLIPS = 2


def _noise_percent_to_flips(noise_percent: int, input_bits: int = 64) -> int:
    bounded_percent = max(0, min(100, int(noise_percent)))
    flips = int(round((bounded_percent / 100.0) * input_bits))
    return max(0, min(input_bits, flips))


def _model_display_name(model_type: ModelType) -> str:
    return DISPLAY_NAME.get(model_type, model_type)


def _ordered_model_types() -> Sequence[ModelType]:
    return [model_type for model_type in DISPLAY_ORDER if model_type in MODEL_TYPES]


@dataclass
class TrainingReport:
    model_type: ModelType
    epochs: int
    augment_per_symbol: int
    samples_seen: int
    clean_accuracy: float
    noisy_accuracy: float
    model_path: Path
    noise_label: str


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
    noise_percent: int | None = None,
) -> TrainingReport:
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if augment_per_symbol < 0:
        raise ValueError("augment_per_symbol must be >= 0")

    rng = random.Random(seed)
    base_vectors = _build_training_samples(pattern_dir)

    if noise_percent is not None:
        flips_per_sample = _noise_percent_to_flips(noise_percent)
        noise_label = f"{noise_percent}%Noise"
        noisy_metric_label = f"{noise_percent}%Noisy"
    else:
        noise_label = f"{flips_per_sample}FlipsNoise"
        noisy_metric_label = "Noisy %"

    model = create_initial_model(
        model_type=model_type,
        pattern_dir=pattern_dir,
        vigilance=vigilance,
        learning_rate=learning_rate,
        choice_alpha=choice_alpha,
    )

    samples_seen = 0
    use_augmentation = model_type == "aug_fuz_ART"
    single_pass = model_type == "ART_sing"

    # Single-pass models see each pattern exactly once in a fixed order.
    effective_epochs = 1 if single_pass else epochs

    for _ in range(effective_epochs):
        labels = list(ALPHABET_A_TO_T)
        if not single_pass:
            rng.shuffle(labels)
        for label in labels:
            base_vector = base_vectors[label]
            model.train_pattern(base_vector, label, augmented=False)
            samples_seen += 1

            if use_augmentation:
                for _ in range(augment_per_symbol):
                    augmented = _flip_bits(base_vector, DEFAULT_AUGMENTED_TRAINING_FLIPS, rng)
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
        noisy_metric_label=noisy_metric_label,
    )
    model.save(model_path)

    return TrainingReport(
        model_type=model_type,
        epochs=effective_epochs,
        augment_per_symbol=augment_per_symbol if use_augmentation else 0,
        samples_seen=samples_seen,
        clean_accuracy=clean_accuracy,
        noisy_accuracy=noisy_accuracy,
        model_path=model_path,
        noise_label=noise_label,
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
    noise_percent: int | None = None,
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
            noise_percent=noise_percent,
        )
        reports[model_type] = report

    return reports


def _format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"


def _print_report(report: TrainingReport) -> None:
    print(f"Model type: {_model_display_name(report.model_type)}")
    print(f"Model saved to: {report.model_path}")
    print(f"Epochs: {report.epochs}")
    print(f"Augment/sample per symbol: {report.augment_per_symbol}")
    print(f"Samples seen: {report.samples_seen}")
    print(f"Clean accuracy: {_format_accuracy(report.clean_accuracy)}")
    print(f"{report.noise_label} accuracy: {_format_accuracy(report.noisy_accuracy)}")


def _print_comparison_table(reports: Dict[ModelType, TrainingReport]) -> None:
    if reports:
        noise_label = next(iter(reports.values())).noise_label
    else:
        noise_label = "Noisy"
    print("\nComparison")
    print(f"Model           Clean      {noise_label:<11}Samples")
    print("------------------------------------------------")

    for model_type in _ordered_model_types():
        if model_type not in reports:
            continue
        report = reports[model_type]
        print(
            f"{_model_display_name(report.model_type):<14}"
            f"{_format_accuracy(report.clean_accuracy):<11}"
            f"{_format_accuracy(report.noisy_accuracy):<11}"
            f"{report.samples_seen}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ART models for A-T pattern recognition")
    parser.add_argument("--pattern-dir", type=Path, default=Path("patterns_orig"))
    parser.add_argument("--model-dir", type=Path, default=Path("patterns_Ass4"))
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        choices=[
            "all",
            *MODEL_TYPES,
            "art1",
            "art1_single_pass",
            "fuzzy_art",
            "aug_fuzzy_art",
        ],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--augment-per-symbol", type=int, default=10)
    parser.add_argument("--noise-percent", type=int, default=3)
    parser.add_argument("--seed", type=int, default=498)
    parser.add_argument("--vigilance", type=float, default=0.82)
    parser.add_argument("--learning-rate", type=float, default=0.6)
    parser.add_argument("--choice-alpha", type=float, default=1e-3)

    args = parser.parse_args()

    if args.noise_percent is not None and (args.noise_percent < 0 or args.noise_percent > 100):
        raise ValueError("--noise-percent must be between 0 and 100")

    if args.model_type == "all":
        selected_types: Sequence[ModelType] = MODEL_TYPES
    else:
        selected_types = [normalize_model_type(args.model_type)]

    selected_types = [model_type for model_type in _ordered_model_types() if model_type in selected_types]

    reports = train_models(
        model_types=selected_types,
        pattern_dir=args.pattern_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        augment_per_symbol=args.augment_per_symbol,
        noise_percent=args.noise_percent,
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
        for model_type in _ordered_model_types():
            if model_type in reports:
                print("\n---")
                _print_report(reports[model_type])
        _print_comparison_table(reports)


if __name__ == "__main__":
    main()
