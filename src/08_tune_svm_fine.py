"""Ajuste fino de SVM RBF para GestureBot con evaluación LOSO.

Busca alrededor de la mejor configuración obtenida previamente:
    C=10, gamma=0.003

Uso:
    python src/08_tune_svm_fine.py

Guarda:
    models/gesture_model_svm_fine.joblib
    reports/svm_fine_loso_results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASS_ORDER = ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")

# Búsqueda concentrada alrededor de C=10 y gamma=0.003.
C_VALUES = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
GAMMA_VALUES = (0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajuste fino de SVM RBF con Leave-One-Subject-Out."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/gesture_dataset.npz"),
    )
    return parser.parse_args()


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
        raise ValueError("X contiene NaN o valores infinitos.")

    return result


def build_model(c_value: float, gamma_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma=gamma_value,
                    class_weight="balanced",
                    cache_size=2048,
                ),
            ),
        ]
    )


def evaluate_loso(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    subjects: list[str],
    labels: list[str],
) -> dict[str, Any]:
    predictions = np.empty(len(y), dtype=object)
    folds: dict[str, dict[str, float | int]] = {}

    started = time.perf_counter()

    for held_out in subjects:
        test_mask = subject == held_out
        train_mask = ~test_mask

        fold_model = clone(model)
        fold_model.fit(X[train_mask], y[train_mask])
        fold_predictions = fold_model.predict(X[test_mask]).astype(str)
        predictions[test_mask] = fold_predictions

        folds[held_out] = {
            "samples": int(test_mask.sum()),
            "accuracy": float(
                accuracy_score(y[test_mask], fold_predictions)
            ),
            "macro_f1": float(
                f1_score(
                    y[test_mask],
                    fold_predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    predictions = predictions.astype(str)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "seconds": float(time.perf_counter() - started),
        "folds": folds,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = (
        args.dataset
        if args.dataset.is_absolute()
        else project_root / args.dataset
    )

    data = load_dataset(dataset_path)
    X = data["X"]
    y = data["y"]
    subject = data["subject"]
    feature_names = data["feature_names"]

    subjects = sorted(np.unique(subject).tolist())
    labels = [label for label in CLASS_ORDER if label in set(y.tolist())]

    if len(subjects) < 2:
        raise ValueError("LOSO necesita al menos dos personas.")

    candidates = [
        (c_value, gamma_value)
        for c_value in C_VALUES
        for gamma_value in GAMMA_VALUES
    ]

    print("=" * 78)
    print("GESTUREBOT — AJUSTE FINO SVM RBF CON LOSO")
    print("=" * 78)
    print(
        f"Muestras: {len(y)} | Variables: {X.shape[1]} | "
        f"Personas: {len(subjects)}"
    )
    print(f"Configuraciones: {len(candidates)}")
    print()

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for number, (c_value, gamma_value) in enumerate(candidates, start=1):
        model = build_model(c_value, gamma_value)
        metrics = evaluate_loso(
            model, X, y, subject, subjects, labels
        )

        result = {
            "C": c_value,
            "gamma": gamma_value,
            **metrics,
        }
        results.append(result)

        marker = ""
        if best is None or (
            result["macro_f1"],
            result["accuracy"],
        ) > (
            best["macro_f1"],
            best["accuracy"],
        ):
            best = result
            marker = "  <-- MEJOR"

        print(
            f"[{number:02d}/{len(candidates):02d}] "
            f"C={c_value:<5g} gamma={gamma_value:<6g} | "
            f"accuracy={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"{metrics['seconds']:.1f}s{marker}"
        )

    assert best is not None

    print()
    print("=" * 78)
    print("MEJOR CONFIGURACIÓN")
    print("=" * 78)
    print(f"C={best['C']} | gamma={best['gamma']}")
    print(f"Accuracy LOSO: {best['accuracy']:.4f}")
    print(f"Macro F1 LOSO: {best['macro_f1']:.4f}")

    final_model = build_model(best["C"], best["gamma"])
    final_model.fit(X, y)

    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "gesture_model_svm_fine.joblib"
    joblib.dump(
        {
            "model": final_model,
            "model_name": "svm_rbf_fine",
            "class_names": np.asarray(labels, dtype=str),
            "feature_names": feature_names,
            "number_of_features": int(X.shape[1]),
            "normalization_version": "wrist_origin_palm_scale_v1",
            "validation_strategy": "leave_one_subject_out",
            "trained_subjects": np.asarray(subjects, dtype=str),
            "parameters": {
                "C": best["C"],
                "gamma": best["gamma"],
            },
        },
        model_path,
    )

    report_path = reports_dir / "svm_fine_loso_results.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best": best,
                "all_results": results,
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
            "El ajuste fino no ha alcanzado el 97 %. "
            "El siguiente paso debe ser revisar los errores por persona/gesto."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
