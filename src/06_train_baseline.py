"""Entrenamiento y evaluación LOSO para GestureBot.

Evalúa los modelos dejando fuera una persona completa en cada iteración
(Leave-One-Subject-Out). Después selecciona el mejor modelo por Macro F1 y
lo entrena con todo el dataset.

No necesita pandas.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CLASS_ORDER = ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")
MODEL_NAMES = ("logistic_regression", "random_forest", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara modelos mediante Leave-One-Subject-Out."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/gesture_dataset.npz"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="Modelos que se entrenarán.",
    )
    parser.add_argument(
        "--exclude-subject",
        action="append",
        default=[],
        help="Persona que se excluirá. Se puede repetir.",
    )
    return parser.parse_args()


def resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el dataset: {path}\n"
            "Ejecuta antes: python src/05_prepare_dataset.py"
        )

    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "subject", "session", "feature_names"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Faltan arrays en el NPZ: {sorted(missing)}")
        result = {name: np.asarray(data[name]) for name in required}

    result["X"] = result["X"].astype(np.float32)
    result["y"] = result["y"].astype(str)
    result["subject"] = result["subject"].astype(str)
    result["session"] = result["session"].astype(str)
    result["feature_names"] = result["feature_names"].astype(str)

    if result["X"].ndim != 2:
        raise ValueError(f"X debe ser bidimensional y tiene forma {result['X'].shape}.")
    if len(result["X"]) != len(result["y"]):
        raise ValueError("X e y no tienen el mismo número de muestras.")
    if len(result["subject"]) != len(result["y"]):
        raise ValueError("subject e y no tienen el mismo número de muestras.")
    if not np.all(np.isfinite(result["X"])):
        raise ValueError("X contiene valores NaN o infinitos.")

    return result


def exclude_subjects(
    data: dict[str, np.ndarray], excluded_subjects: list[str]
) -> dict[str, np.ndarray]:
    excluded = {value.strip().lower() for value in excluded_subjects if value.strip()}
    if not excluded:
        return data

    lowered = np.char.lower(data["subject"])
    keep = ~np.isin(lowered, list(excluded))

    missing = sorted(excluded - set(lowered.tolist()))
    if missing:
        print("AVISO: no se encontraron para excluir:", ", ".join(missing))

    if not np.any(keep):
        raise ValueError("Se han excluido todas las muestras.")

    filtered: dict[str, np.ndarray] = {}
    for key, value in data.items():
        filtered[key] = value if key == "feature_names" else value[keep]
    return filtered


def get_labels(y: np.ndarray) -> list[str]:
    found = set(y.tolist())
    unexpected = sorted(found - set(CLASS_ORDER))
    if unexpected:
        raise ValueError(f"Etiquetas inesperadas: {unexpected}")

    labels = [label for label in CLASS_ORDER if label in found]
    if len(labels) < 2:
        raise ValueError("Se necesitan al menos dos clases.")
    return labels


def validate_subjects(
    y: np.ndarray, subjects_array: np.ndarray, labels: list[str]
) -> list[str]:
    subjects = sorted(np.unique(subjects_array).tolist())
    if len(subjects) < 2:
        raise ValueError("LOSO necesita al menos dos personas distintas.")

    problems: list[str] = []
    for person in subjects:
        person_labels = set(y[subjects_array == person].tolist())
        missing = [label for label in labels if label not in person_labels]
        if missing:
            problems.append(f"{person}: {', '.join(missing)}")

    if problems:
        raise ValueError(
            "Faltan gestos para algunas personas:\n  - "
            + "\n  - ".join(problems)
        )
    return subjects


def print_distribution(
    y: np.ndarray,
    subjects_array: np.ndarray,
    subjects: list[str],
    labels: list[str],
) -> None:
    print("\nMUESTRAS POR PERSONA")
    print("-" * 78)
    print(f"{'Persona':<16}" + "".join(f"{label:>11}" for label in labels) + f"{'TOTAL':>10}")
    print("-" * 78)

    for person in subjects:
        counts = Counter(y[subjects_array == person].tolist())
        total = sum(counts.values())
        print(
            f"{person:<16}"
            + "".join(f"{counts[label]:>11}" for label in labels)
            + f"{total:>10}"
        )
    print("-" * 78)


def build_models(selected: list[str]) -> dict[str, Any]:
    available: dict[str, Any] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(96, 48),
                        batch_size=64,
                        max_iter=400,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=25,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
    return {name: available[name] for name in selected}


def save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicción",
        ylabel="Clase real",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison(results: dict[str, dict[str, Any]], path: Path) -> None:
    names = list(results)
    accuracy = [results[name]["accuracy"] for name in names]
    macro_f1 = [results[name]["macro_f1"] for name in names]

    positions = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - width / 2, accuracy, width, label="Accuracy")
    ax.bar(positions + width / 2, macro_f1, width, label="Macro F1")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Puntuación")
    ax.set_title("Comparación LOSO")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    model_name: str,
    base_model: Any,
    X: np.ndarray,
    y: np.ndarray,
    subjects_array: np.ndarray,
    subjects: list[str],
    labels: list[str],
    figures_dir: Path,
) -> dict[str, Any]:
    all_predictions = np.empty(len(y), dtype=object)
    fold_results: dict[str, dict[str, float | int]] = {}
    total_training_seconds = 0.0
    total_inference_seconds = 0.0

    print(f"\n{'=' * 72}")
    print(f"MODELO: {model_name}")
    print("=" * 72)

    for position, held_out in enumerate(subjects, start=1):
        test_mask = subjects_array == held_out
        train_mask = ~test_mask

        model = clone(base_model)

        start = time.perf_counter()
        model.fit(X[train_mask], y[train_mask])
        training_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = model.predict(X[test_mask]).astype(str)
        inference_seconds = time.perf_counter() - start

        all_predictions[test_mask] = predictions

        accuracy = accuracy_score(y[test_mask], predictions)
        macro_f1 = f1_score(
            y[test_mask],
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        fold_results[held_out] = {
            "test_samples": int(test_mask.sum()),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "training_seconds": float(training_seconds),
            "mean_inference_ms_per_sample": float(
                inference_seconds / max(int(test_mask.sum()), 1) * 1000
            ),
        }

        total_training_seconds += training_seconds
        total_inference_seconds += inference_seconds

        print(
            f"[{position:02d}/{len(subjects):02d}] "
            f"Persona de prueba: {held_out:<15} "
            f"| accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f}"
        )

    all_predictions = all_predictions.astype(str)

    accuracy = accuracy_score(y, all_predictions)
    macro_f1 = f1_score(
        y,
        all_predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        y,
        all_predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y, all_predictions, labels=labels)

    matrix_path = figures_dir / f"confusion_matrix_loso_{model_name}.png"
    save_confusion_matrix(
        matrix,
        labels,
        matrix_path,
        f"Matriz de confusión LOSO — {model_name}",
    )

    print("-" * 72)
    print(f"Accuracy LOSO global: {accuracy:.4f}")
    print(f"Macro F1 LOSO global: {macro_f1:.4f}")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "total_training_seconds": float(total_training_seconds),
        "mean_inference_ms_per_sample": float(
            total_inference_seconds / max(len(y), 1) * 1000
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "folds": fold_results,
        "confusion_matrix_path": str(matrix_path),
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = resolve(project_root, args.dataset)

    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = load_dataset(dataset_path)
        data = exclude_subjects(data, args.exclude_subject)

        X = data["X"]
        y = data["y"]
        subjects_array = data["subject"]
        feature_names = data["feature_names"]

        labels = get_labels(y)
        subjects = validate_subjects(y, subjects_array, labels)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR:\n{error}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("GESTUREBOT — LEAVE-ONE-SUBJECT-OUT")
    print("=" * 72)
    print(f"Dataset: {dataset_path}")
    print(f"Muestras: {len(y)}")
    print(f"Variables: {X.shape[1]}")
    print(f"Personas: {len(subjects)}")
    print(f"Modelos: {', '.join(args.models)}")

    print_distribution(y, subjects_array, subjects, labels)

    models = build_models(args.models)
    results: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        results[model_name] = evaluate_model(
            model_name,
            model,
            X,
            y,
            subjects_array,
            subjects,
            labels,
            figures_dir,
        )

    best_name = max(
        results,
        key=lambda name: (results[name]["macro_f1"], results[name]["accuracy"]),
    )

    print(f"\nMejor modelo: {best_name}")
    print("Entrenándolo de nuevo con todas las personas...")

    final_model = clone(models[best_name])
    final_model.fit(X, y)

    model_path = models_dir / "gesture_model.joblib"
    joblib.dump(
        {
            "model": final_model,
            "model_name": best_name,
            "class_names": np.asarray(labels, dtype=str),
            "feature_names": feature_names,
            "number_of_features": int(X.shape[1]),
            "normalization_version": "wrist_origin_palm_scale_v1",
            "validation_strategy": "leave_one_subject_out",
            "trained_subjects": np.asarray(subjects, dtype=str),
        },
        model_path,
    )

    comparison_path = figures_dir / "model_comparison_loso.png"
    save_comparison(results, comparison_path)

    metadata = {
        "best_model": best_name,
        "selection_metric": "macro_f1_loso",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "number_of_samples": int(len(y)),
        "number_of_features": int(X.shape[1]),
        "subjects": subjects,
        "classes": labels,
        "excluded_subjects": args.exclude_subject,
        "metrics": {
            name: {
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "total_training_seconds": result["total_training_seconds"],
                "mean_inference_ms_per_sample": result[
                    "mean_inference_ms_per_sample"
                ],
            }
            for name, result in results.items()
        },
    }

    with (models_dir / "training_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    with (reports_dir / "loso_results.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"metadata": metadata, "models": results},
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 72)
    print(f"Mejor modelo: {best_name}")
    print(f"Accuracy LOSO: {results[best_name]['accuracy']:.4f}")
    print(f"Macro F1 LOSO: {results[best_name]['macro_f1']:.4f}")
    print(f"Modelo guardado en: {model_path}")
    print(f"Resultados guardados en: {reports_dir / 'loso_results.json'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
