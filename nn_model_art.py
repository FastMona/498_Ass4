"""ART model utilities for 8x8 alphabet recognition (A-T)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

from PIL import Image


ALPHABET_A_TO_T: Tuple[str, ...] = tuple(chr(code) for code in range(ord("A"), ord("T") + 1))
DEFAULT_PATTERN_DIR = Path("patterns_orig")
DEFAULT_MODEL_DIR = Path("patterns_Ass4")
ModelType = Literal["art1", "fuzzy_art", "aug_fuzzy_art"]
MODEL_TYPES: Tuple[ModelType, ...] = ("art1", "fuzzy_art", "aug_fuzzy_art")


def default_model_path(model_type: ModelType) -> Path:
    return DEFAULT_MODEL_DIR / f"{model_type}_a_to_t.json"


def _threshold_pixel(value: int) -> float:
    return 1.0 if value < 128 else 0.0


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def load_pattern_vector(image_path: Path, expected_size: Tuple[int, int] = (8, 8)) -> List[float]:
    """Load and flatten a pattern image as a binary vector of length 64."""
    image = Image.open(image_path).convert("L")
    if image.size != expected_size:
        image = image.resize(expected_size, Image.Resampling.NEAREST)
    pixels = list(image.tobytes())
    return [_threshold_pixel(px) for px in pixels]


def discover_pattern_images(pattern_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not pattern_dir.exists():
        return mapping

    for label in ALPHABET_A_TO_T:
        for extension in (".png", ".jpg", ".jpeg"):
            candidate = pattern_dir / f"pattern_{label}{extension}"
            if candidate.exists():
                mapping[label] = candidate
                break

    return mapping


def fuzzy_and_sum(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(min(x, y) for x, y in zip(a, b))


def complement_code(vector: Sequence[float]) -> List[float]:
    return list(vector) + [1.0 - value for value in vector]


@dataclass
class Prediction:
    label: str
    confidence: float


class BaseARTCharacterModel:
    """Base class for ART-family character models."""

    model_type: ModelType = "fuzzy_art"

    def __init__(
        self,
        vigilance: float = 0.82,
        learning_rate: float = 0.6,
        choice_alpha: float = 1e-3,
        allowed_labels: Iterable[str] = ALPHABET_A_TO_T,
        input_size: int = 64,
    ) -> None:
        self.vigilance = float(vigilance)
        self.learning_rate = float(learning_rate)
        self.choice_alpha = float(choice_alpha)
        self.allowed_labels = tuple(sorted(set(allowed_labels)))
        self.input_size = int(input_size)
        self.templates: List[List[float]] = []
        self.template_labels: List[str] = []
        self.training_metrics: Dict[str, object] = {}

    @property
    def coded_size(self) -> int:
        return self.input_size

    def _preprocess_input(self, vector: Sequence[float]) -> List[float]:
        return [_clip01(float(value)) for value in vector]

    def _encode_input(self, vector: Sequence[float]) -> List[float]:
        return list(vector)

    def _choice(self, coded_input: Sequence[float], template: Sequence[float]) -> float:
        numerator = fuzzy_and_sum(coded_input, template)
        denominator = self.choice_alpha + sum(template)
        return numerator / denominator

    def _match(self, coded_input: Sequence[float], template: Sequence[float]) -> float:
        numerator = fuzzy_and_sum(coded_input, template)
        denominator = sum(coded_input) + 1e-12
        return numerator / denominator

    def _update_template(self, category_index: int, coded_input: Sequence[float]) -> None:
        current = self.templates[category_index]
        updated: List[float] = []
        for value, weight in zip(coded_input, current):
            fuzzy_value = min(value, weight)
            new_weight = self.learning_rate * fuzzy_value + (1.0 - self.learning_rate) * weight
            updated.append(new_weight)
        self.templates[category_index] = updated

    def _train_with_vigilance(self, vector: Sequence[float], label: str, vigilance: float) -> int:
        if label not in self.allowed_labels:
            raise ValueError(f"Label '{label}' is not in allowed labels: {self.allowed_labels}")
        if len(vector) != self.input_size:
            raise ValueError(f"Expected vector length {self.input_size}, got {len(vector)}")

        preprocessed = self._preprocess_input(vector)
        coded_input = self._encode_input(preprocessed)

        candidates = [
            (self._choice(coded_input, template), index)
            for index, template_label in enumerate(self.template_labels)
            if template_label == label
            for template in [self.templates[index]]
        ]
        candidates.sort(reverse=True)

        for _, category_index in candidates:
            template = self.templates[category_index]
            if self._match(coded_input, template) >= vigilance:
                self._update_template(category_index, coded_input)
                return category_index

        self.templates.append(list(coded_input))
        self.template_labels.append(label)
        return len(self.templates) - 1

    def train_pattern(self, vector: Sequence[float], label: str, augmented: bool = False) -> int:
        _ = augmented
        return self._train_with_vigilance(vector, label, self.vigilance)

    def predict(self, vector: Sequence[float]) -> Prediction:
        if not self.templates:
            raise RuntimeError("Model has no templates. Train or load a model first.")
        if len(vector) != self.input_size:
            raise ValueError(f"Expected vector length {self.input_size}, got {len(vector)}")

        preprocessed = self._preprocess_input(vector)
        coded_input = self._encode_input(preprocessed)

        best_index = 0
        best_choice = float("-inf")
        for idx, template in enumerate(self.templates):
            score = self._choice(coded_input, template)
            if score > best_choice:
                best_choice = score
                best_index = idx

        match_score = self._match(coded_input, self.templates[best_index])
        return Prediction(label=self.template_labels[best_index], confidence=match_score)

    def summary(self) -> Dict[str, object]:
        by_label: Dict[str, int] = {label: 0 for label in self.allowed_labels}
        for label in self.template_labels:
            by_label[label] = by_label.get(label, 0) + 1

        summary = {
            "model_type": self.model_type,
            "vigilance": self.vigilance,
            "learning_rate": self.learning_rate,
            "choice_alpha": self.choice_alpha,
            "input_size": self.input_size,
            "coded_size": self.coded_size,
            "templates": len(self.templates),
            "templates_per_label": by_label,
        }
        summary.update(self.training_metrics)
        return summary

    def set_training_metrics(self, clean_accuracy: float, noisy_accuracy: float, samples_seen: int) -> None:
        self.training_metrics = {
            "clean_accuracy": float(clean_accuracy),
            "noisy_accuracy": float(noisy_accuracy),
            "samples_seen": int(samples_seen),
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": self.model_type,
            "vigilance": self.vigilance,
            "learning_rate": self.learning_rate,
            "choice_alpha": self.choice_alpha,
            "input_size": self.input_size,
            "allowed_labels": list(self.allowed_labels),
            "templates": self.templates,
            "template_labels": self.template_labels,
            "training_metrics": self.training_metrics,
        }

    @staticmethod
    def _as_float(value: object, default: float) -> float:
        if isinstance(value, (int, float, str)):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def _as_int(value: object, default: int) -> int:
        if isinstance(value, (int, float, str)):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def _as_label_list(value: object, default: Sequence[str]) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, tuple):
            return [str(item) for item in value]
        return [str(item) for item in default]

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "BaseARTCharacterModel":
        vigilance = cls._as_float(payload.get("vigilance"), 0.82)
        learning_rate = cls._as_float(payload.get("learning_rate"), 0.6)
        choice_alpha = cls._as_float(payload.get("choice_alpha"), 1e-3)
        allowed_labels = cls._as_label_list(payload.get("allowed_labels"), ALPHABET_A_TO_T)
        input_size = cls._as_int(payload.get("input_size"), 64)

        model = cls(
            vigilance=vigilance,
            learning_rate=learning_rate,
            choice_alpha=choice_alpha,
            allowed_labels=allowed_labels,
            input_size=input_size,
        )

        templates_raw = payload.get("templates")
        labels_raw = payload.get("template_labels")

        parsed_templates: List[List[float]] = []
        if isinstance(templates_raw, list):
            for row in templates_raw:
                if isinstance(row, list):
                    parsed_templates.append([cls._as_float(value, 0.0) for value in row])

        parsed_labels = cls._as_label_list(labels_raw, [])

        model.templates = parsed_templates
        model.template_labels = parsed_labels

        training_metrics = payload.get("training_metrics")
        if isinstance(training_metrics, dict):
            model.training_metrics = dict(training_metrics)

        if len(model.templates) != len(model.template_labels):
            raise ValueError("Saved model is invalid: templates and labels length mismatch")

        return model

    def save(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


class ART1CharacterModel(BaseARTCharacterModel):
    model_type: ModelType = "art1"

    def _preprocess_input(self, vector: Sequence[float]) -> List[float]:
        return [1.0 if float(value) >= 0.5 else 0.0 for value in vector]

    def _update_template(self, category_index: int, coded_input: Sequence[float]) -> None:
        current = self.templates[category_index]
        self.templates[category_index] = [
            1.0 if current_value > 0.5 and input_value > 0.5 else 0.0
            for current_value, input_value in zip(current, coded_input)
        ]


class FuzzyARTCharacterModel(BaseARTCharacterModel):
    model_type: ModelType = "fuzzy_art"

    @property
    def coded_size(self) -> int:
        return self.input_size * 2

    def _encode_input(self, vector: Sequence[float]) -> List[float]:
        return complement_code(vector)


class AugmentedFuzzyARTCharacterModel(FuzzyARTCharacterModel):
    model_type: ModelType = "aug_fuzzy_art"

    def __init__(
        self,
        vigilance: float = 0.82,
        learning_rate: float = 0.6,
        choice_alpha: float = 1e-3,
        allowed_labels: Iterable[str] = ALPHABET_A_TO_T,
        input_size: int = 64,
        augmented_vigilance_drop: float = 0.08,
        min_vigilance: float = 0.65,
    ) -> None:
        super().__init__(
            vigilance=vigilance,
            learning_rate=learning_rate,
            choice_alpha=choice_alpha,
            allowed_labels=allowed_labels,
            input_size=input_size,
        )
        self.augmented_vigilance_drop = float(augmented_vigilance_drop)
        self.min_vigilance = float(min_vigilance)

    def train_pattern(self, vector: Sequence[float], label: str, augmented: bool = False) -> int:
        if augmented:
            effective_vigilance = max(self.min_vigilance, self.vigilance - self.augmented_vigilance_drop)
            return self._train_with_vigilance(vector, label, effective_vigilance)
        return self._train_with_vigilance(vector, label, self.vigilance)

    def to_dict(self) -> Dict[str, object]:
        payload = super().to_dict()
        payload["augmented_vigilance_drop"] = self.augmented_vigilance_drop
        payload["min_vigilance"] = self.min_vigilance
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "AugmentedFuzzyARTCharacterModel":
        vigilance = cls._as_float(payload.get("vigilance"), 0.82)
        learning_rate = cls._as_float(payload.get("learning_rate"), 0.6)
        choice_alpha = cls._as_float(payload.get("choice_alpha"), 1e-3)
        allowed_labels = cls._as_label_list(payload.get("allowed_labels"), ALPHABET_A_TO_T)
        input_size = cls._as_int(payload.get("input_size"), 64)

        model = cls(
            vigilance=vigilance,
            learning_rate=learning_rate,
            choice_alpha=choice_alpha,
            allowed_labels=allowed_labels,
            input_size=input_size,
            augmented_vigilance_drop=cls._as_float(payload.get("augmented_vigilance_drop"), 0.08),
            min_vigilance=cls._as_float(payload.get("min_vigilance"), 0.65),
        )

        templates_raw = payload.get("templates")
        labels_raw = payload.get("template_labels")

        parsed_templates: List[List[float]] = []
        if isinstance(templates_raw, list):
            for row in templates_raw:
                if isinstance(row, list):
                    parsed_templates.append([cls._as_float(value, 0.0) for value in row])

        parsed_labels = cls._as_label_list(labels_raw, [])

        model.templates = parsed_templates
        model.template_labels = parsed_labels

        training_metrics = payload.get("training_metrics")
        if isinstance(training_metrics, dict):
            model.training_metrics = dict(training_metrics)

        if len(model.templates) != len(model.template_labels):
            raise ValueError("Saved model is invalid: templates and labels length mismatch")

        return model


def create_model(
    model_type: ModelType,
    vigilance: float = 0.82,
    learning_rate: float = 0.6,
    choice_alpha: float = 1e-3,
) -> BaseARTCharacterModel:
    if model_type == "art1":
        return ART1CharacterModel(vigilance=vigilance, learning_rate=learning_rate, choice_alpha=choice_alpha)
    if model_type == "fuzzy_art":
        return FuzzyARTCharacterModel(vigilance=vigilance, learning_rate=learning_rate, choice_alpha=choice_alpha)
    if model_type == "aug_fuzzy_art":
        return AugmentedFuzzyARTCharacterModel(
            vigilance=vigilance,
            learning_rate=learning_rate,
            choice_alpha=choice_alpha,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def create_initial_model(
    model_type: ModelType,
    pattern_dir: Path = DEFAULT_PATTERN_DIR,
    vigilance: float = 0.82,
    learning_rate: float = 0.6,
    choice_alpha: float = 1e-3,
) -> BaseARTCharacterModel:
    image_map = discover_pattern_images(pattern_dir)
    missing = [label for label in ALPHABET_A_TO_T if label not in image_map]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing pattern files for labels: {missing_text}. Expected files in '{pattern_dir}'."
        )

    model = create_model(
        model_type=model_type,
        vigilance=vigilance,
        learning_rate=learning_rate,
        choice_alpha=choice_alpha,
    )

    for label in ALPHABET_A_TO_T:
        vector = load_pattern_vector(image_map[label])
        model.train_pattern(vector, label)

    return model


def create_initial_models(
    model_types: Sequence[ModelType],
    pattern_dir: Path = DEFAULT_PATTERN_DIR,
    vigilance: float = 0.82,
    learning_rate: float = 0.6,
    choice_alpha: float = 1e-3,
) -> Dict[ModelType, BaseARTCharacterModel]:
    models: Dict[ModelType, BaseARTCharacterModel] = {}
    for model_type in model_types:
        models[model_type] = create_initial_model(
            model_type=model_type,
            pattern_dir=pattern_dir,
            vigilance=vigilance,
            learning_rate=learning_rate,
            choice_alpha=choice_alpha,
        )
    return models


def load_model(model_path: Path) -> BaseARTCharacterModel:
    with model_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Saved model must be a JSON object")

    model_type_raw = payload.get("model_type", "fuzzy_art")
    model_type = str(model_type_raw)
    if model_type == "art1":
        return ART1CharacterModel.from_dict(payload)
    if model_type == "fuzzy_art":
        return FuzzyARTCharacterModel.from_dict(payload)
    if model_type == "aug_fuzzy_art":
        return AugmentedFuzzyARTCharacterModel.from_dict(payload)

    raise ValueError(f"Unknown model type in model file: {model_type}")


def load_vector_from_path(image_path: Path) -> List[float]:
    return load_pattern_vector(image_path)
