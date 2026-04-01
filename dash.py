"""Run the assignment workflow from one menu-driven dashboard."""

from __future__ import annotations

import builtins
import io
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import patterns as pattern_manager
from noise import run_create_noisy_patterns
from nn_model_art import (
    ALPHABET_A_TO_T,
    MODEL_TYPES,
    ModelType,
    default_model_path,
    discover_pattern_images,
    load_model,
    load_pattern_vector,
    load_vector_from_path,
    resolve_model_path,
)
from nn_train_art import train_models


LAST_SWEEP_LOW = 0.60
LAST_SWEEP_HIGH = 0.95

DISPLAY_ORDER: Tuple[ModelType, ...] = (
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

_last_paths: Dict[str, Path] = {}
_path_memory_file = Path("path_memory.json")


def _is_valid_path_text(value: str) -> bool:
    if not value:
        return False
    if "Operation failed:" in value:
        return False
    # Guard against terminal control bytes being treated as file paths.
    for char in value:
        if ord(char) < 32 and char not in ("\t",):
            return False
    return True


def _load_path_memory() -> None:
    global _last_paths
    if not _path_memory_file.exists():
        return
    try:
        raw = json.loads(_path_memory_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            loaded: Dict[str, Path] = {}
            changed = False
            for key, value in raw.items():
                if isinstance(key, str) and isinstance(value, str) and _is_valid_path_text(value):
                    loaded[key] = Path(value)
                else:
                    changed = True
            _last_paths = loaded
            if changed:
                _save_path_memory()
    except Exception:
        # Keep startup resilient when saved path memory is invalid.
        pass


def _save_path_memory() -> None:
    payload = {key: str(value) for key, value in _last_paths.items()}
    _path_memory_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class _TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _install_terminal_capture() -> Tuple[io.StringIO, object, object, object]:
    capture = io.StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_input = builtins.input

    sys.stdout = _TeeStream(sys.stdout, capture)
    sys.stderr = _TeeStream(sys.stderr, capture)

    def _logged_input(prompt: str = "") -> str:
        response = original_input(prompt)
        capture.write(f"{response}\n")
        return response

    builtins.input = _logged_input
    return capture, original_stdout, original_stderr, original_input


def _append_terminal_capture(capture_text: str) -> None:
    log_path = Path("terminal.txt")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'=' * 72}\n")
        log_file.write(f"Session exit: {stamp}\n")
        log_file.write(f"{'=' * 72}\n")
        log_file.write(capture_text)
        if not capture_text.endswith("\n"):
            log_file.write("\n")


def _uninstall_terminal_capture(original_stdout, original_stderr, original_input) -> None:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    builtins.input = original_input


def _print_header() -> None:
    print("\n" + "=" * 72)
    print("ART Dashboard - Alphabet A to T")
    print("=" * 72)


def _model_display_name(model_type: ModelType) -> str:
    return DISPLAY_NAME.get(model_type, model_type)


def _ordered_model_types() -> Sequence[ModelType]:
    return [model_type for model_type in DISPLAY_ORDER if model_type in MODEL_TYPES]


def _ask_path(prompt: str, default: Path) -> Path:
    current = _last_paths.get(prompt, default)
    raw = input(f"{prompt} (Enter to keep: {current}): ").strip()
    if raw and not _is_valid_path_text(raw):
        print("Invalid path input detected; keeping current value.")
        result = current
    else:
        result = Path(raw) if raw else current
    _last_paths[prompt] = result
    _save_path_memory()
    return result


def _select_model_types() -> Sequence[ModelType]:
    print("\nSelect model type")
    print("1. ART_1")
    print("2. ART_sing")
    print("3. fuzzy_ART")
    print("4. aug_fuz_ART")
    print("5. All four (default)")

    raw = input("Choose an option (1-5, default 5): ").strip() or "5"
    if raw == "1":
        return ["ART_1"]
    if raw == "2":
        return ["ART_sing"]
    if raw == "3":
        return ["fuzzy_ART"]
    if raw == "4":
        return ["aug_fuz_ART"]
    return list(_ordered_model_types())


def _select_single_model_type(default: ModelType = "fuzzy_ART") -> ModelType:
    print("\nSelect trained model")
    print("1. ART_1")
    print("2. ART_sing")
    print("3. fuzzy_ART")
    print("4. aug_fuz_ART")

    default_option = "3"
    if default == "ART_1":
        default_option = "1"
    elif default == "ART_sing":
        default_option = "2"
    elif default == "aug_fuz_ART":
        default_option = "4"

    raw = input(f"Choose an option (1-4, default {default_option}): ").strip() or default_option
    if raw == "1":
        return "ART_1"
    if raw == "2":
        return "ART_sing"
    if raw == "3":
        return "fuzzy_ART"
    if raw == "4":
        return "aug_fuz_ART"
    return default


def _flip_bits(vector: Sequence[float], flips: int, rng: random.Random) -> List[float]:
    augmented = list(vector)
    flips = max(0, min(flips, len(augmented)))
    for index in rng.sample(range(len(augmented)), flips):
        augmented[index] = 0.0 if augmented[index] > 0.5 else 1.0
    return augmented


def _noise_percent_to_flips(noise_percent: int, input_bits: int = 64) -> int:
    bounded_percent = max(0, min(100, int(noise_percent)))
    flips = int(round((bounded_percent / 100.0) * input_bits))
    return max(0, min(input_bits, flips))


def _noise_percent_from_model_summary(summary: Dict[str, object], default_percent: int = 3) -> int:
    label_value = summary.get("noisy_metric_label")
    if isinstance(label_value, str):
        match = re.search(r"(\d+)\s*%", label_value)
        if match:
            return max(0, min(100, int(match.group(1))))
    return default_percent


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

    print("\nVigilance sweep across trained models")
    model_dir = _ask_path("Model directory", Path("patterns_Ass4"))
    pattern_dir = _ask_path("Pattern directory for evaluation", Path("patterns_orig"))

    loaded_models: Dict[ModelType, object] = {}
    for model_type in _ordered_model_types():
        model_path = resolve_model_path(model_dir, model_type)
        if model_path.exists():
            loaded_models[model_type] = load_model(model_path)

    if not loaded_models:
        print(f"No model files found in '{model_dir}'.")
        return

    low_raw = input(f"Low vigilance (default {LAST_SWEEP_LOW:.2f}): ").strip() or f"{LAST_SWEEP_LOW:.2f}"
    high_raw = input(f"High vigilance (default {LAST_SWEEP_HIGH:.2f}): ").strip() or f"{LAST_SWEEP_HIGH:.2f}"
    low = float(low_raw)
    high = float(high_raw)

    if low > high:
        low, high = high, low

    LAST_SWEEP_LOW = low
    LAST_SWEEP_HIGH = high

    reference_type = next(iter(loaded_models.keys()))
    reference_summary = loaded_models[reference_type].summary()
    noise_percent = _noise_percent_from_model_summary(reference_summary, default_percent=3)
    flips_per_sample = _noise_percent_to_flips(noise_percent)

    clean_samples, noisy_samples = _build_eval_sets(
        pattern_dir=pattern_dir,
        flips_per_sample=flips_per_sample,
    )

    step = (high - low) / 9.0 if high != low else 0.0
    levels = [low + step * idx for idx in range(10)]

    present_model_types = [model_type for model_type in _ordered_model_types() if model_type in loaded_models]

    rows: List[Dict[str, str]] = []
    for level in levels:
        row: Dict[str, str] = {"Vigilance": f"{level:.3f}"}
        for model_type in present_model_types:
            model = loaded_models[model_type]
            clean_acc, _ = _score_at_vigilance(model, clean_samples, level)
            noisy_acc, _ = _score_at_vigilance(model, noisy_samples, level)
            row[f"clean_{model_type}"] = f"{clean_acc * 100:.0f}%"
            row[f"noisy_{model_type}"] = f"{noisy_acc * 100:.0f}%"
        rows.append(row)

    vig_col_w = max(len("Vigilance"), max(len(row["Vigilance"]) for row in rows))
    model_sub_col_w: Dict[ModelType, int] = {}
    model_block_w: Dict[ModelType, int] = {}
    for model_type in present_model_types:
        display = _model_display_name(model_type)
        width = 1
        for row in rows:
            width = max(width, len(row[f"clean_{model_type}"]))
            width = max(width, len(row[f"noisy_{model_type}"]))
        model_sub_col_w[model_type] = width
        model_block_w[model_type] = max(len(display), width * 2 + 1)

    header_line_top = "  ".join(
        [f"{'':>{vig_col_w}}"]
        + [f"{_model_display_name(model_type):^{model_block_w[model_type]}}" for model_type in present_model_types]
    )
    header_line_bottom = "  ".join(
        [f"{'Vigilance':>{vig_col_w}}"]
        + [
            f"{'Clean':>{model_sub_col_w[model_type]}} {'Noisy':>{model_sub_col_w[model_type]}}"
            for model_type in present_model_types
        ]
    )
    rule = "=" * max(len(header_line_top), len(header_line_bottom))

    print("\n VIGILANCE SWEEP RESULTS")
    print(f" Models: {', '.join(_model_display_name(model_type) for model_type in present_model_types)}")
    print(f" Noise:  {noise_percent}% ({flips_per_sample} bit flips)")
    print(rule)
    print(header_line_top)
    print(header_line_bottom)
    print(rule)
    for row in rows:
        model_cells = [
            f"{row[f'clean_{model_type}']:>{model_sub_col_w[model_type]}} {row[f'noisy_{model_type}']:>{model_sub_col_w[model_type]}}"
            for model_type in present_model_types
        ]
        print("  ".join([f"{row['Vigilance']:>{vig_col_w}}"] + model_cells))
    print(rule)


def manage_patterns() -> None:
    selected_folder = _ask_path("Pattern folder for manager", Path("patterns_orig"))

    # Reuse patterns.py without duplicating its menu logic.
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
    noise_percent_raw = input("Noise % 0-100 (default 3): ").strip() or "3"
    try:
        noise_percent = int(noise_percent_raw)
    except ValueError:
        print("Invalid noise percent. Enter an integer from 0 to 100.")
        return

    if noise_percent < 0 or noise_percent > 100:
        print("Invalid noise percent. Enter an integer from 0 to 100.")
        return

    flips_per_sample = _noise_percent_to_flips(noise_percent)
    noise_label = f"{noise_percent}%Noise"

    print("\nNote: augmentation is applied only to aug_fuz_ART.")
    print("Augmented training for aug_fuz_ART always flips exactly 2 bits per synthetic sample.")
    print(f"Noise setting for noisy evaluation/reporting: {noise_label} ({flips_per_sample} bit flips on 8x8 input)")

    reports = train_models(
        model_types=model_types,
        pattern_dir=pattern_dir,
        model_dir=model_dir,
        epochs=epochs,
        augment_per_symbol=augment_per_symbol,
        flips_per_sample=flips_per_sample,
        noise_percent=noise_percent,
        vigilance=float(vigilance_raw),
        learning_rate=float(learning_raw),
        choice_alpha=float(alpha_raw),
    )

    print("\nCreate + train complete")
    print(f"Model           Clean      {noise_label:<11}Samples")
    print("------------------------------------------------")

    for model_type in _ordered_model_types():
        if model_type not in reports:
            continue
        report = reports[model_type]
        print(
            f"{_model_display_name(report.model_type):<14}"
            f"{report.clean_accuracy * 100:>6.2f}%   "
            f"{report.noisy_accuracy * 100:>6.2f}%   "
            f"{report.samples_seen}"
        )
        print(f"Saved: {report.model_path}")


def recognize_image() -> None:
    model_dir = _ask_path("Model directory", Path("patterns_Ass4"))
    image_path = _ask_path("Image path to recognize", Path("patterns_orig") / "pattern_A.jpg")

    if not image_path.exists():
        print(f"Image file does not exist: {image_path}")
        return

    vector = load_vector_from_path(image_path)

    rows: List[Tuple[str, str, str]] = []
    for model_type in _ordered_model_types():
        model_path = resolve_model_path(model_dir, model_type)
        if not model_path.exists():
            rows.append((_model_display_name(model_type), "-", "-"))
            continue
        model = load_model(model_path)
        prediction = model.predict(vector)
        rows.append((_model_display_name(model_type), prediction.label, f"{prediction.confidence * 100:.2f}%"))

    model_col_w = max(len("Model"), max(len(r[0]) for r in rows))
    pred_col_w = max(len("Predicted"), max(len(r[1]) for r in rows))
    conf_col_w = max(len("Confidence"), max(len(r[2]) for r in rows))

    header_line = (
        f"{'Model':<{model_col_w}}  "
        f"{'Predicted':<{pred_col_w}}  "
        f"{'Confidence':>{conf_col_w}}"
    )
    rule = "=" * len(header_line)

    print(f"\n{rule}")
    print(" SINGLE CHAR RECOG RESULTS")
    print(f" Model dir: {model_dir}")
    print(f" Image:     {image_path}")
    print(rule)
    print(header_line)
    print(rule)
    for model_type, label, confidence in rows:
        print(f"{model_type:<{model_col_w}}  {label:<{pred_col_w}}  {confidence:>{conf_col_w}}")
    print(rule)


def recognize_folder() -> None:
    model_dir = _ask_path("Model directory", Path("patterns_Ass4"))
    image_dir = _ask_path("Image folder to recognize", Path("patterns_orig"))
    monte_carlo_raw = input("Number of Monte Carlo runs (default 10): ").strip() or "10"

    try:
        monte_carlo_runs = int(monte_carlo_raw)
    except ValueError:
        print("Invalid run count. Enter an integer >= 1.")
        return

    if monte_carlo_runs < 1:
        print("Invalid run count. Enter an integer >= 1.")
        return

    image_map = discover_pattern_images(image_dir)
    if not image_map:
        print(f"No pattern images found in '{image_dir}'.")
        return

    labels_found = sorted(image_map.keys())

    # Load each model once to keep batch evaluation responsive.
    models = {}
    for model_type in MODEL_TYPES:
        model_path = resolve_model_path(model_dir, model_type)
        if model_path.exists():
            models[model_type] = load_model(model_path)

    if not models:
        print(f"No model files found in '{model_dir}'.")
        return

    # Build table content first so column widths fit every value.
    ordered_types = list(_ordered_model_types())
    recognized_folder_name = image_dir.name if image_dir.name else str(image_dir)
    model_vigilance_text_parts: List[str] = []
    for model_type in ordered_types:
        if model_type not in models:
            continue
        summary = models[model_type].summary()
        vigilance_value = summary.get("vigilance")
        if isinstance(vigilance_value, (int, float, str)):
            model_vigilance_text_parts.append(f"{_model_display_name(model_type)}={float(vigilance_value):.3f}")
    vigilance_header_text = ", ".join(model_vigilance_text_parts) if model_vigilance_text_parts else "N/A"

    correct_counts: Dict[str, int] = {mt: 0 for mt in ordered_types}
    per_label_hits: Dict[str, int] = {}
    predictions: Dict[str, Dict[str, str]] = {}
    for label in labels_found:
        vector = load_pattern_vector(image_map[label])
        predictions[label] = {}
        label_hits = 0
        for model_type in ordered_types:
            if model_type not in models:
                predictions[label][model_type] = "N/A"
            else:
                pred = models[model_type].predict(vector)
                cell = f"{pred.label} ({pred.confidence * 100:.1f}%)"
                if pred.label == label:
                    correct_counts[model_type] += 1
                    label_hits += 1
                predictions[label][model_type] = cell
        per_label_hits[label] = label_hits

    total = len(labels_found)

    acc_cells: Dict[str, str] = {}
    for model_type in ordered_types:
        if model_type in models and total:
            acc_cells[model_type] = f"{correct_counts[model_type]}/{total} ({correct_counts[model_type] / total * 100:.1f}%)"
        else:
            acc_cells[model_type] = "N/A"

    # Widths adapt to confidence strings and summary values.
    star_cells = {label: "*" * per_label_hits[label] for label in labels_found}
    label_col_w = max(len("Label"), len("Acc"))
    star_col_w = max(len("*"), max((len(star_cells[label]) for label in labels_found), default=0))
    col_widths: Dict[str, int] = {}
    for model_type in ordered_types:
        all_vals = [predictions[lbl][model_type] for lbl in labels_found] + [acc_cells[model_type]]
        display_name = _model_display_name(model_type)
        col_widths[model_type] = max(len(display_name), max(len(v) for v in all_vals))

    def _build_row(left: str, stars: str, cells: Dict[str, str]) -> str:
        parts = [f"{left:<{label_col_w}}", f"{stars:<{star_col_w}}"]
        for mt in ordered_types:
            parts.append(f"{cells[mt]:>{col_widths[mt]}}")
        return "  ".join(parts)

    header_line = _build_row("Label", "*", {mt: _model_display_name(mt) for mt in ordered_types})
    rule = "=" * len(header_line)

    print(f"\n{rule}")
    print(" BATCH RECOGNITION RESULTS")
    print(f" Training: {model_dir}")
    print(f" Recog:    {image_dir}")
    print(f" Folder:   {recognized_folder_name}")
    print(f" Vig:      {vigilance_header_text}")
    print(rule)
    print(header_line)
    print(rule)

    for label in labels_found:
        print(_build_row(label, star_cells[label], predictions[label]))

    print(rule)
    print(_build_row("Acc", "", acc_cells))
    print(rule)

    if monte_carlo_runs == 1:
        return

    def _safe_ratio(numerator: int, denominator: int) -> float:
        return (numerator / denominator) if denominator else 0.0

    def _mean_with_ci95(values: Sequence[float]) -> Tuple[float, float]:
        mean_value = sum(values) / len(values)
        if len(values) < 2:
            return mean_value, 0.0
        variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
        std_dev = math.sqrt(max(0.0, variance))
        ci_half_width = 1.96 * std_dev / math.sqrt(len(values))
        return mean_value, ci_half_width

    metric_runs: Dict[str, Dict[str, List[float]]] = {
        model_type: {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for model_type in ordered_types
        if model_type in models
    }

    class_labels = list(labels_found)

    for _ in range(monte_carlo_runs):
        expected_labels: List[str] = []
        predicted_by_model: Dict[str, List[str]] = {model_type: [] for model_type in metric_runs}

        for label in labels_found:
            vector = load_pattern_vector(image_map[label])
            expected_labels.append(label)
            for model_type in metric_runs:
                prediction = models[model_type].predict(vector)
                predicted_by_model[model_type].append(prediction.label)

        sample_count = len(expected_labels)
        for model_type, predicted_labels in predicted_by_model.items():
            correct = sum(1 for expected, predicted in zip(expected_labels, predicted_labels) if expected == predicted)
            accuracy = _safe_ratio(correct, sample_count)

            per_class_precision: List[float] = []
            per_class_recall: List[float] = []
            per_class_f1: List[float] = []
            for class_label in class_labels:
                true_positive = sum(
                    1
                    for expected, predicted in zip(expected_labels, predicted_labels)
                    if expected == class_label and predicted == class_label
                )
                false_positive = sum(
                    1
                    for expected, predicted in zip(expected_labels, predicted_labels)
                    if expected != class_label and predicted == class_label
                )
                false_negative = sum(
                    1
                    for expected, predicted in zip(expected_labels, predicted_labels)
                    if expected == class_label and predicted != class_label
                )

                precision = _safe_ratio(true_positive, true_positive + false_positive)
                recall = _safe_ratio(true_positive, true_positive + false_negative)
                f1 = _safe_ratio(2 * precision * recall, precision + recall)

                per_class_precision.append(precision)
                per_class_recall.append(recall)
                per_class_f1.append(f1)

            macro_precision = sum(per_class_precision) / len(per_class_precision)
            macro_recall = sum(per_class_recall) / len(per_class_recall)
            macro_f1 = sum(per_class_f1) / len(per_class_f1)

            metric_runs[model_type]["accuracy"].append(accuracy)
            metric_runs[model_type]["precision"].append(macro_precision)
            metric_runs[model_type]["recall"].append(macro_recall)
            metric_runs[model_type]["f1"].append(macro_f1)

    def _metric_text(values: Sequence[float], as_percent: bool) -> str:
        mean_value, ci_half_width = _mean_with_ci95(values)
        if as_percent:
            return f"{mean_value * 100:.2f}% +- {ci_half_width * 100:.2f}%"
        return f"{mean_value:.4f} +- {ci_half_width:.4f}"

    summary_rows: List[Dict[str, str]] = []
    for model_type in ordered_types:
        if model_type not in metric_runs:
            continue
        summary_rows.append(
            {
                "Model": _model_display_name(model_type),
                "Accuracy": _metric_text(metric_runs[model_type]["accuracy"], as_percent=True),
                "Precision": _metric_text(metric_runs[model_type]["precision"], as_percent=False),
                "Recall": _metric_text(metric_runs[model_type]["recall"], as_percent=False),
                "F1": _metric_text(metric_runs[model_type]["f1"], as_percent=False),
            }
        )

    columns = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    col_widths: Dict[str, int] = {}
    for column in columns:
        col_widths[column] = max(len(column), max(len(row[column]) for row in summary_rows))

    summary_header = "  ".join(f"{column:>{col_widths[column]}}" for column in columns)
    summary_rule = "=" * len(summary_header)

    print(f"\n{summary_rule}")
    print(" MONTE CARLO METRIC SUMMARY")
    print(f" Runs: {monte_carlo_runs}")
    print(f" Folder: {recognized_folder_name}")
    print(f" Vig: {vigilance_header_text}")
    print(" Metrics: mean +- 95% CI")
    print(summary_rule)
    print(summary_header)
    print(summary_rule)
    for row in summary_rows:
        print("  ".join(f"{row[column]:>{col_widths[column]}}" for column in columns))
    print(summary_rule)


def show_model_summary() -> None:
    model_dir = _ask_path("Model directory", Path("patterns_Ass4"))

    noisy_metric_name = "Noisy %"
    metric_names = ["Clean %", noisy_metric_name, "Samples", "Vig", "LR", "alpha", "Templates"] + list(ALPHABET_A_TO_T)
    model_values: Dict[str, List[str]] = {}
    ordered_types = list(_ordered_model_types())
    for model_type in ordered_types:
        model_path = resolve_model_path(model_dir, model_type)
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
        noisy_metric_label_value = summary.get("noisy_metric_label")
        samples_seen_value = summary.get("samples_seen")

        if isinstance(noisy_metric_label_value, str) and noisy_metric_label_value.strip():
            noisy_metric_name = noisy_metric_label_value.strip()

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

    metric_names = ["Clean %", noisy_metric_name, "Samples", "Vig", "LR", "alpha", "Templates"] + list(ALPHABET_A_TO_T)

    print("\n MODEL TRAINING TABLE")
    model_headers = ordered_types
    row_header_width = max(len("Metric"), max(len(name) for name in metric_names))

    col_widths: List[int] = []
    for model_name in model_headers:
        values = model_values.get(model_name, ["-"] * len(metric_names))
        col_width = max(len(_model_display_name(model_name)), max(len(value) for value in values))
        col_widths.append(col_width)

    header_cells = [f"{'Metric':<{row_header_width}}"]
    for model_name, width in zip(model_headers, col_widths):
        header_cells.append(f"{_model_display_name(model_name):>{width}}")
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
        print("2. Create + train ART models (ART_sing / ART_1 / fuzzy_ART / aug_fuz_ART)")
        print("3. Sweep vigilance on one trained model")
        print("4. Recognize a character image")
        print("5. Recognize a folder (batch of images)")
        print("6. Show model summary")
        print("7. Noisy patterns")
        print("8. (not implemented)")
        print("9. Exit")

        choice = input("Choose an option (1-9): ").strip()

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
                recognize_folder()
            elif choice == "6":
                show_model_summary()
            elif choice == "7":
                run_create_noisy_patterns()
            elif choice == "8":
                print("Not implemented yet.")
            elif choice == "9":
                print("Goodbye.")
                break
            else:
                print("Invalid choice. Enter 1-9.")
        except Exception as exc:
            print(f"Operation failed: {exc}")


if __name__ == "__main__":
    _load_path_memory()
    _capture, _orig_stdout, _orig_stderr, _orig_input = _install_terminal_capture()
    try:
        main()
    finally:
        _uninstall_terminal_capture(_orig_stdout, _orig_stderr, _orig_input)
        _append_terminal_capture(_capture.getvalue())
