"""Dashboard for comparing ART-family models on A-T recognition."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import patterns as pattern_manager
from nn_model_art import (
    ALPHABET_A_TO_T,
    MODEL_TYPES,
    ModelType,
    default_model_path,
    discover_pattern_images,
    load_model,
    load_pattern_vector,
    load_vector_from_path,
)
from nn_train_art import train_models


LAST_SWEEP_LOW = 0.60
LAST_SWEEP_HIGH = 0.95

_last_paths: Dict[str, Path] = {}


def _print_header() -> None:
    print("\n" + "=" * 72)
    print("ART Dashboard - Alphabet A to T")
    print("=" * 72)


def _ask_path(prompt: str, default: Path) -> Path:
    current = _last_paths.get(prompt, default)
    raw = input(f"{prompt} (Enter to keep: {current}): ").strip()
    result = Path(raw) if raw else current
    _last_paths[prompt] = result
    return result


def _select_model_types() -> Sequence[ModelType]:
    print("\nSelect model type")
    print("1. ART1")
    print("2. Fuzzy ART")
    print("3. Augmented Fuzzy ART")
    print("4. All three (default)")

    raw = input("Choose an option (1-4, default 4): ").strip() or "4"
    if raw == "1":
        return ["art1"]
    if raw == "2":
        return ["fuzzy_art"]
    if raw == "3":
        return ["aug_fuzzy_art"]
    return list(MODEL_TYPES)


def _select_single_model_type(default: ModelType = "fuzzy_art") -> ModelType:
    print("\nSelect trained model")
    print("1. ART1")
    print("2. Fuzzy ART")
    print("3. Augmented Fuzzy ART")

    default_option = "2"
    if default == "art1":
        default_option = "1"
    elif default == "aug_fuzzy_art":
        default_option = "3"

    raw = input(f"Choose an option (1-3, default {default_option}): ").strip() or default_option
    if raw == "1":
        return "art1"
    if raw == "2":
        return "fuzzy_art"
    if raw == "3":
        return "aug_fuzzy_art"
    return default


def _flip_bits(vector: Sequence[float], flips: int, rng: random.Random) -> List[float]:
    augmented = list(vector)
    flips = max(0, min(flips, len(augmented)))
    for index in rng.sample(range(len(augmented)), flips):
        augmented[index] = 0.0 if augmented[index] > 0.5 else 1.0
    return augmented


def _build_eval_sets(
    pattern_dir: Path,
    noisy_per_label: int = 10,
    flips_per_sample: int = 2,
    seed: int = 498,
) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]]]:
    image_map = discover_pattern_images(pattern_dir)
    missing = [label for label in ALPHABET_A_TO_T if label not in image_map]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing pattern files for labels: {missing_text}. Expected files in '{pattern_dir}'."
        )

    clean: Dict[str, List[List[float]]] = {}
    noisy: Dict[str, List[List[float]]] = {}
    rng = random.Random(seed)

    for label in ALPHABET_A_TO_T:
        base_vector = load_pattern_vector(image_map[label])
        clean[label] = [base_vector]
        noisy[label] = [_flip_bits(base_vector, flips_per_sample, rng) for _ in range(noisy_per_label)]

    return clean, noisy


def _score_at_vigilance(
    model,
    samples: Dict[str, List[List[float]]],
    vigilance: float,
) -> Tuple[float, float]:
    total = 0
    accepted = 0
    correct = 0

    for expected_label, vectors in samples.items():
        for vector in vectors:
            prediction = model.predict(vector)
            total += 1
            if prediction.confidence >= vigilance:
                accepted += 1
                if prediction.label == expected_label:
                    correct += 1

    accuracy = (correct / accepted) if accepted else 0.0
    coverage = (accepted / total) if total else 0.0
    return accuracy, coverage


def vigilance_sweep() -> None:
    global LAST_SWEEP_LOW, LAST_SWEEP_HIGH

    print("\nVigilance sweep on one trained model")
    model_type = _select_single_model_type(default="fuzzy_art")
    model_path = Path("patterns_Ass4") / default_model_path(model_type).name
    pattern_dir = _ask_path("Pattern directory for evaluation", Path("patterns_orig"))

    if not model_path.exists():
        print(f"Model file does not exist: {model_path}")
        return

    low_raw = input(f"Low vigilance (default {LAST_SWEEP_LOW:.2f}): ").strip() or f"{LAST_SWEEP_LOW:.2f}"
    high_raw = input(f"High vigilance (default {LAST_SWEEP_HIGH:.2f}): ").strip() or f"{LAST_SWEEP_HIGH:.2f}"
    low = float(low_raw)
    high = float(high_raw)

    if low > high:
        low, high = high, low

    LAST_SWEEP_LOW = low
    LAST_SWEEP_HIGH = high

    clean_samples, noisy_samples = _build_eval_sets(pattern_dir=pattern_dir)
    model = load_model(model_path)

    step = (high - low) / 9.0 if high != low else 0.0
    levels = [low + step * idx for idx in range(10)]

    print("\nModel type:", model.summary()["model_type"])
    print("Vigilance sweep results (10 levels)")
    print("Level  Vigilance  CleanAcc  NoisyAcc  OverallAcc  Coverage")
    print("-------------------------------------------------------------")

    for idx, level in enumerate(levels, start=1):
        clean_acc, clean_cov = _score_at_vigilance(model, clean_samples, level)
        noisy_acc, noisy_cov = _score_at_vigilance(model, noisy_samples, level)

        total_cov = (clean_cov + noisy_cov) / 2.0

        # Overall accuracy computed over accepted clean+noisy samples.
        combined_samples: Dict[str, List[List[float]]] = {}
        for label in ALPHABET_A_TO_T:
            combined_samples[label] = clean_samples[label] + noisy_samples[label]
        overall_acc, _ = _score_at_vigilance(model, combined_samples, level)

        print(
            f"{idx:>2}     {level:>7.3f}   "
            f"{clean_acc * 100:>7.2f}%  "
            f"{noisy_acc * 100:>7.2f}%  "
            f"{overall_acc * 100:>9.2f}%  "
            f"{total_cov * 100:>7.2f}%"
        )


def manage_patterns() -> None:
    selected_folder = _ask_path("Pattern folder for manager", Path("patterns_orig"))

    # Configure patterns.py defaults so its menu operations target the selected folder.
    pattern_manager.DEFAULT_OUTPUT_DIR = str(selected_folder).replace("\\", "/")
    pattern_manager.PATTERN_DB = pattern_manager.BASE_DIR / pattern_manager.DEFAULT_OUTPUT_DIR / "patterns"

    print(f"\nOpening pattern manager for folder: {selected_folder}")
    pattern_manager.main()


def create_art_models() -> None:
    pattern_dir = _ask_path("Pattern directory", Path("patterns_orig"))
    model_dir = _ask_path("Model directory", Path("patterns_Ass4"))
    model_types = _select_model_types()

    vigilance_raw = input("Vigilance 0-1 (default 0.82): ").strip() or "0.82"
    learning_raw = input("Learning rate 0-1 (default 0.6): ").strip() or "0.6"
    alpha_raw = input("Choice alpha (default 0.001): ").strip() or "0.001"

    epochs = int(input("Epochs (default 20): ").strip() or "20")
    augment_per_symbol = int(input("Augmentations per symbol per epoch (default 10): ").strip() or "10")
    flips_per_sample = int(input("Bit flips per augmentation (default 2): ").strip() or "2")

    print("\nNote: augmentation is applied only to aug_fuzzy_art.")

    reports = train_models(
        model_types=model_types,
        pattern_dir=pattern_dir,
        model_dir=model_dir,
        epochs=epochs,
        augment_per_symbol=augment_per_symbol,
        flips_per_sample=flips_per_sample,
        vigilance=float(vigilance_raw),
        learning_rate=float(learning_raw),
        choice_alpha=float(alpha_raw),
    )

    print("\nCreate + train complete")
    print("Model           Clean      Noisy      Samples")
    print("------------------------------------------------")

    sorted_reports = sorted(reports.values(), key=lambda item: item.noisy_accuracy, reverse=True)
    for report in sorted_reports:
        print(
            f"{report.model_type:<14}"
            f"{report.clean_accuracy * 100:>6.2f}%   "
            f"{report.noisy_accuracy * 100:>6.2f}%   "
            f"{report.samples_seen}"
        )
        print(f"Saved: {report.model_path}")


def recognize_image() -> None:
    model_dir = _ask_path("Model directory", Path("patterns_Ass4"))
    image_path = _ask_path("Image path to recognize", Path("patterns_orig") / "pattern_A.png")

    if not image_path.exists():
        print(f"Image file does not exist: {image_path}")
        return

    vector = load_vector_from_path(image_path)

    rows: List[Tuple[str, str, str]] = []
    for model_type in MODEL_TYPES:
        model_path = model_dir / default_model_path(model_type).name
        if not model_path.exists():
            rows.append((model_type, "-", "-"))
            continue
        model = load_model(model_path)
        prediction = model.predict(vector)
        rows.append((model_type, prediction.label, f"{prediction.confidence * 100:.2f}%"))

    print("\nRecognition results")
    print(f"Image: {image_path}")
    print(f"{'Model':<18}  {'Predicted':<10}  {'Confidence':>10}")
    print("-" * 44)
    for model_type, label, confidence in rows:
        print(f"{model_type:<18}  {label:<10}  {confidence:>10}")


def show_model_summary() -> None:
    model_dir = _ask_path("Model directory", Path("patterns_Ass4"))

    metric_names = ["Clean %", "Noisy %", "Samples", "Vig", "LR", "alpha", "Templates"] + list(ALPHABET_A_TO_T)
    model_values: Dict[str, List[str]] = {}
    for model_type in MODEL_TYPES:
        model_path = model_dir / default_model_path(model_type).name
        if not model_path.exists():
            model_values[model_type] = ["-", "-", "-", "-", "-", "-", "-"] + ["-"] * len(ALPHABET_A_TO_T)
            continue

        model = load_model(model_path)
        summary = model.summary()

        vig_value = summary.get("vigilance")
        lr_value = summary.get("learning_rate")
        alpha_value = summary.get("choice_alpha")
        clean_accuracy_value = summary.get("clean_accuracy")
        noisy_accuracy_value = summary.get("noisy_accuracy")
        samples_seen_value = summary.get("samples_seen")

        clean_accuracy_text = "-"
        if isinstance(clean_accuracy_value, (int, float, str)):
            clean_accuracy_text = f"{float(clean_accuracy_value) * 100:.2f}%"

        noisy_accuracy_text = "-"
        if isinstance(noisy_accuracy_value, (int, float, str)):
            noisy_accuracy_text = f"{float(noisy_accuracy_value) * 100:.2f}%"

        samples_seen_text = "-"
        if isinstance(samples_seen_value, (int, float, str)):
            samples_seen_text = str(int(float(samples_seen_value)))

        vig_text = "-"
        if isinstance(vig_value, (int, float, str)):
            vig_text = f"{float(vig_value):.3f}"

        lr_text = "-"
        if isinstance(lr_value, (int, float, str)):
            lr_text = f"{float(lr_value):.3f}"

        alpha_text = "-"
        if isinstance(alpha_value, (int, float, str)):
            alpha_text = f"{float(alpha_value):.6f}"

        templates_value = summary.get("templates")
        templates_text = "-"
        if isinstance(templates_value, (int, float, str)):
            templates_text = str(int(float(templates_value)))

        per_label = summary.get("templates_per_label")
        per_label_counts: List[str] = []
        for label in ALPHABET_A_TO_T:
            count_text = "-"
            if isinstance(per_label, dict):
                raw_count = per_label.get(label)
                if isinstance(raw_count, (int, float, str)):
                    count_text = str(int(float(raw_count)))
            per_label_counts.append(count_text)

        model_values[model_type] = [
            clean_accuracy_text,
            noisy_accuracy_text,
            samples_seen_text,
            vig_text,
            lr_text,
            alpha_text,
            templates_text,
        ] + per_label_counts

    print("\n MODEL TRAINING TABLE")
    model_headers = list(MODEL_TYPES)
    row_header_width = max(len("Metric"), max(len(name) for name in metric_names))

    col_widths: List[int] = []
    for model_name in model_headers:
        values = model_values.get(model_name, ["-"] * len(metric_names))
        col_width = max(len(model_name), max(len(value) for value in values))
        col_widths.append(col_width)

    header_cells = [f"{'Metric':<{row_header_width}}"]
    for model_name, width in zip(model_headers, col_widths):
        header_cells.append(f"{model_name:>{width}}")
    header_line = "  ".join(header_cells)
    print("=" * len(header_line))
    print(header_line)
    print("=" * len(header_line))

    for index, metric_name in enumerate(metric_names):
        row_cells = [f"{metric_name:<{row_header_width}}"]
        for model_name, width in zip(model_headers, col_widths):
            values = model_values.get(model_name, ["-"] * len(metric_names))
            row_cells.append(f"{values[index]:>{width}}")
        print("  ".join(row_cells))
    print("=" * len(header_line))


def main() -> None:
    while True:
        _print_header()
        print("1. Manage patterns (patterns.py)")
        print("2. Create + train ART models (ART1 / Fuzzy / Aug Fuzzy)")
        print("3. Sweep vigilance on one trained model")
        print("4. Recognize a character image")
        print("5. Show model summary")
        print("6. Exit")

        choice = input("Choose an option (1-6): ").strip()

        try:
            if choice == "1":
                manage_patterns()
            elif choice == "2":
                create_art_models()
            elif choice == "3":
                vigilance_sweep()
            elif choice == "4":
                recognize_image()
            elif choice == "5":
                show_model_summary()
            elif choice == "6":
                print("Goodbye.")
                break
            else:
                print("Invalid choice. Enter 1, 2, 3, 4, 5, or 6.")
        except Exception as exc:
            print(f"Operation failed: {exc}")


if __name__ == "__main__":
    main()
