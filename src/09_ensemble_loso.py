"""Ensemble LOSO para GestureBot.

Combina:
- Regresión logística
- Random Forest
- SVM RBF ajustada (C=6, gamma=0.0015)

Entrena cada modelo una vez por fold LOSO, guarda sus probabilidades y prueba
varias ponderaciones sin volver a entrenar.

Uso:
    python src/09_ensemble_loso.py

Salidas:
    models/gesture_model_ensemble.joblib
    reports/ensemble_loso_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASS_ORDER = ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")
RANDOM_STATE = 42

WEIGHT_SETS = {
    "equal": (1.0, 1.0, 1.0),
    "svm_x2": (1.0, 1.0, 2.0),
    "svm_x3": (1.0, 1.0, 3.0),
    "logreg_svm": (1.0, 0.0, 2.0),
    "rf_svm": (0.0, 1.0, 2.0),
    "logreg_x2_svm_x3": (2.0, 1.0, 3.0),
    "rf_x2_svm_x3": (1.0, 2.0, 3.0),
}


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el dataset: {path}\n"
            "Ejecuta antes: python src/05_prepare_dataset.py"
        )

    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "subject", "feature_names"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Faltan arrays en el NPZ: {sorted(missing)}")
        result = {name: np.asarray(data[name]) for name in required}

    result["X"] = result["X"].astype(np.float32)
    result["y"] = result["y"].astype(str)
    result["subject"] = result["subject"].astype(str)
    result["feature_names"] = result["feature_names"].astype(str)

    if result["X"].ndim != 2:
        raise ValueError(f"X tiene una forma inválida: {result['X'].shape}")
    if len(result["X"]) != len(result["y"]):
        raise ValueError("X e y no tienen el mismo número de muestras.")
    if len(result["subject"]) != len(result["y"]):
        raise ValueError("subject e y no tienen el mismo número de muestras.")
    if not np.all(np.isfinite(result["X"])):
        raise ValueError("X contiene NaN o infinitos.")

    return result


def build_models() -> dict[str, Any]:
    return {
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
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=6.0,
                        gamma=0.0015,
                        class_weight="balanced",
                        probability=True,
                        cache_size=2048,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def aligned_probabilities(
    model: Any,
    X: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    probabilities = model.predict_proba(X)
    model_classes = [str(value) for value in model.classes_]

    aligned = np.zeros((len(X), len(labels)), dtype=np.float64)
    for target_index, label in enumerate(labels):
        if label not in model_classes:
            raise ValueError(f"El modelo no contiene la clase {label}.")
        source_index = model_classes.index(label)
        aligned[:, target_index] = probabilities[:, source_index]

    return aligned


def metrics_from_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> tuple[float, float, np.ndarray]:
    prediction_indices = np.argmax(probabilities, axis=1)
    predictions = np.asarray([labels[index] for index in prediction_indices], dtype=str)

    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(
        y_true,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return float(accuracy), float(macro_f1), predictions


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data/processed/gesture_dataset.npz"

    data = load_dataset(dataset_path)
    X = data["X"]
    y = data["y"]
    subject = data["subject"]
    feature_names = data["feature_names"]

    subjects = sorted(np.unique(subject).tolist())
    labels = [label for label in CLASS_ORDER if label in set(y.tolist())]

    if len(subjects) < 2:
        raise ValueError("LOSO necesita al menos dos personas.")

    probability_store = {
        "logistic_regression": np.zeros((len(y), len(labels)), dtype=np.float64),
        "random_forest": np.zeros((len(y), len(labels)), dtype=np.float64),
        "svm_rbf": np.zeros((len(y), len(labels)), dtype=np.float64),
    }

    fold_info: dict[str, Any] = {}
    base_models = build_models()

    print("=" * 78)
    print("GESTUREBOT — ENSEMBLE CON EVALUACIÓN LOSO")
    print("=" * 78)
    print(f"Muestras: {len(y)} | Variables: {X.shape[1]} | Personas: {len(subjects)}")
    print()

    started = time.perf_counter()

    for fold_number, held_out in enumerate(subjects, start=1):
        test_mask = subject == held_out
        train_mask = ~test_mask

        print(
            f"[{fold_number:02d}/{len(subjects):02d}] "
            f"Persona de prueba: {held_out}"
        )

        fold_info[held_out] = {"samples": int(test_mask.sum()), "models": {}}

        for model_name, base_model in base_models.items():
            model = clone(base_model)

            train_start = time.perf_counter()
            model.fit(X[train_mask], y[train_mask])
            train_seconds = time.perf_counter() - train_start

            probabilities = aligned_probabilities(model, X[test_mask], labels)
            probability_store[model_name][test_mask] = probabilities

            accuracy, macro_f1, _ = metrics_from_probabilities(
                y[test_mask], probabilities, labels
            )
            fold_info[held_out]["models"][model_name] = {
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "training_seconds": train_seconds,
            }

            print(
                f"    {model_name:<22} "
                f"accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f}"
            )

    print()
    print("=" * 78)
    print("COMPARACIÓN DE PONDERACIONES")
    print("=" * 78)

    results: dict[str, Any] = {}

    for name, weights in WEIGHT_SETS.items():
        logistic_weight, forest_weight, svm_weight = weights
        total_weight = logistic_weight + forest_weight + svm_weight

        combined = (
            logistic_weight * probability_store["logistic_regression"]
            + forest_weight * probability_store["random_forest"]
            + svm_weight * probability_store["svm_rbf"]
        ) / total_weight

        accuracy, macro_f1, predictions = metrics_from_probabilities(
            y, combined, labels
        )

        per_subject: dict[str, Any] = {}
        for held_out in subjects:
            mask = subject == held_out
            subject_accuracy = accuracy_score(y[mask], predictions[mask])
            subject_f1 = f1_score(
                y[mask],
                predictions[mask],
                labels=labels,
                average="macro",
                zero_division=0,
            )
            per_subject[held_out] = {
                "accuracy": float(subject_accuracy),
                "macro_f1": float(subject_f1),
            }

        results[name] = {
            "weights": {
                "logistic_regression": logistic_weight,
                "random_forest": forest_weight,
                "svm_rbf": svm_weight,
            },
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_subject": per_subject,
        }

        marker = "  <-- >= 97 %" if accuracy >= 0.97 else ""
        print(
            f"{name:<22} accuracy={accuracy:.4f} | "
            f"macro_f1={macro_f1:.4f}{marker}"
        )

    best_name = max(
        results,
        key=lambda key: (results[key]["macro_f1"], results[key]["accuracy"]),
    )
    best = results[best_name]

    print()
    print("=" * 78)
    print("MEJOR ENSEMBLE")
    print("=" * 78)
    print(f"Configuración: {best_name}")
    print(f"Pesos: {best['weights']}")
    print(f"Accuracy LOSO: {best['accuracy']:.4f}")
    print(f"Macro F1 LOSO: {best['macro_f1']:.4f}")

    print("Entrenando los tres modelos con todo el dataset...")
    final_models = build_models()
    for model in final_models.values():
        model.fit(X, y)

    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "gesture_model_ensemble.joblib"
    joblib.dump(
        {
            "models": final_models,
            "weights": best["weights"],
            "class_names": np.asarray(labels, dtype=str),
            "feature_names": feature_names,
            "number_of_features": int(X.shape[1]),
            "normalization_version": "wrist_origin_palm_scale_v1",
            "validation_strategy": "leave_one_subject_out",
            "trained_subjects": np.asarray(subjects, dtype=str),
        },
        model_path,
    )

    report_path = reports_dir / "ensemble_loso_results.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_name": best_name,
                "best": best,
                "all_results": results,
                "fold_info": fold_info,
                "elapsed_seconds": time.perf_counter() - started,
                "dataset": str(dataset_path),
                "subjects": subjects,
                "labels": labels,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Modelo guardado: {model_path}")
    print(f"Resultados guardados: {report_path}")

    if best["accuracy"] >= 0.97:
        print("OBJETIVO ALCANZADO: accuracy LOSO >= 97 %")
    else:
        print(
            "El ensemble no ha alcanzado el 97 %. "
            "La siguiente mejora debe venir de los datos o de otra representación."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
