"""Run the assignment workflow from one menu-driven dashboard."""

from __future__ import annotations

import builtins
import io
import json
import random
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
)
from nn_train_art import train_models


LAST_SWEEP_LOW = 0.60
LAST_SWEEP_HIGH = 0.95

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
    print("1. ART1")
    print("2. ART1 Single Pass")
    print("3. Fuzzy ART")
    print("4. Augmented Fuzzy ART")
    print("5. All four (default)")

    raw = input("Choose an option (1-5, default 5): ").strip() or "5"
    if raw == "1":
        return ["art1"]
    if raw == "2":
        return ["art1_single_pass"]
    if raw == "3":
        return ["fuzzy_art"]
    if raw == "4":
        return ["aug_fuzzy_art"]
    return list(MODEL_TYPES)


def _select_single_model_type(default: ModelType = "fuzzy_art") -> ModelType:
    print("\nSelect trained model")
    print("1. ART1")
    print("2. ART1 Single Pass")
    print("3. Fuzzy ART")
    print("4. Augmented Fuzzy ART")

    default_option = "3"
    if default == "art1":
        default_option = "1"
    elif default == "art1_single_pass":
        default_option = "2"
    elif default == "aug_fuzzy_art":
        default_option = "4"

    raw = input(f"Choose an option (1-4, default {default_option}): ").strip() or default_option
    if raw == "1":
        return "art1"
    if raw == "2":
        return "art1_single_pass"
    if raw == "3":
        return "fuzzy_art"
    if raw == "4":
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

    rows: List[Dict[str, str]] = []
    for idx, level in enumerate(levels, start=1):
        clean_acc, clean_cov = _score_at_vigilance(model, clean_samples, level)
        noisy_acc, noisy_cov = _score_at_vigilance(model, noisy_samples, level)

        # Report one combined metric so sweep levels are easier to compare.
        combined_samples: Dict[str, List[List[float]]] = {}
        for label in ALPHABET_A_TO_T:
            combined_samples[label] = clean_samples[label] + noisy_samples[label]
        overall_acc, _ = _score_at_vigilance(model, combined_samples, level)

        rows.append(
            {
                "Level": str(idx),
                "Vigilance": f"{level:.3f}",
                "CleanAcc": f"{clean_acc * 100:.2f}%",
                "NoisyAcc": f"{noisy_acc * 100:.2f}%",
                "OverallAcc": f"{overall_acc * 100:.2f}%",
                "CleanCov": f"{clean_cov * 100:.2f}%",
                "NoisyCov": f"{noisy_cov * 100:.2f}%",
            }
        )

    columns = ["Level", "Vigilance", "CleanAcc", "NoisyAcc", "OverallAcc", "CleanCov", "NoisyCov"]
    col_widths: Dict[str, int] = {}
    for col in columns:
        col_widths[col] = max(len(col), max(len(row[col]) for row in rows))

    header_line = "  ".join(f"{col:>{col_widths[col]}}" for col in columns)
    rule = "=" * len(header_line)

    print("\n VIGILANCE SWEEP RESULTS")
    print(f" Model:  {model.summary()['model_type']}")
    print(f" Levels: {len(levels)}")
    print(rule)
    print(header_line)
    print(rule)
    for row in rows:
        print("  ".join(f"{row[col]:>{col_widths[col]}}" for col in columns))
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
    image_path = _ask_path("Image path to recognize", Path("patterns_orig") / "pattern_A.jpg")

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

    image_map = discover_pattern_images(image_dir)
    if not image_map:
        print(f"No pattern images found in '{image_dir}'.")
        return

    labels_found = sorted(image_map.keys())

    # Load each model once to keep batch evaluation responsive.
    models = {}
    for model_type in MODEL_TYPES:
        model_path = model_dir / default_model_path(model_type).name
        if model_path.exists():
            models[model_type] = load_model(model_path)

    if not models:
        print(f"No model files found in '{model_dir}'.")
        return

    # Build table content first so column widths fit every value.
    correct_counts: Dict[str, int] = {mt: 0 for mt in MODEL_TYPES}
    per_label_hits: Dict[str, int] = {}
    predictions: Dict[str, Dict[str, str]] = {}
    for label in labels_found:
        vector = load_pattern_vector(image_map[label])
        predictions[label] = {}
        label_hits = 0
        for model_type in MODEL_TYPES:
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
    for model_type in MODEL_TYPES:
        if model_type in models and total:
            acc_cells[model_type] = f"{correct_counts[model_type]}/{total} ({correct_counts[model_type] / total * 100:.1f}%)"
        else:
            acc_cells[model_type] = "N/A"

    # Widths adapt to confidence strings and summary values.
    star_cells = {label: "*" * per_label_hits[label] for label in labels_found}
    label_col_w = max(len("Label"), len("Acc"))
    star_col_w = max(len("*"), max((len(star_cells[label]) for label in labels_found), default=0))
    col_widths: Dict[str, int] = {}
    for model_type in MODEL_TYPES:
        all_vals = [predictions[lbl][model_type] for lbl in labels_found] + [acc_cells[model_type]]
        col_widths[model_type] = max(len(model_type), max(len(v) for v in all_vals))

    def _build_row(left: str, stars: str, cells: Dict[str, str]) -> str:
        parts = [f"{left:<{label_col_w}}", f"{stars:<{star_col_w}}"]
        for mt in MODEL_TYPES:
            parts.append(f"{cells[mt]:>{col_widths[mt]}}")
        return "  ".join(parts)

    header_line = _build_row("Label", "*", {mt: mt for mt in MODEL_TYPES})
    rule = "=" * len(header_line)

    print(f"\n{rule}")
    print(" BATCH RECOGNITION RESULTS")
    print(f" Training: {model_dir}")
    print(f" Recog:    {image_dir}")
    print(rule)
    print(header_line)
    print(rule)

    for label in labels_found:
        print(_build_row(label, star_cells[label], predictions[label]))

    print(rule)
    print(_build_row("Acc", "", acc_cells))
    print(rule)


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
