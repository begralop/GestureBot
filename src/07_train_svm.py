"""Busca una SVM RBF para GestureBot usando evaluación LOSO.

Uso rápido:
    python src/07_train_svm_loso.py

Búsqueda más amplia:
    python src/07_train_svm_loso.py --full

El script no usa pandas. Guarda el mejor modelo en:
    models/gesture_model_svm.joblib
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASS_ORDER = ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Busca una SVM RBF mediante Leave-One-Subject-Out."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/gesture_dataset.npz"),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Prueba una cuadrícula más amplia y tarda más.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta antes: python src/05_prepare_dataset.py"
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
        raise ValueError(f"X debe ser una matriz; forma recibida: {result['X'].shape}")
    if len(result["X"]) != len(result["y"]):
        raise ValueError("X e y tienen distinto número de muestras.")
    if not np.all(np.isfinite(result["X"])):
        raise ValueError("X contiene NaN o valores infinitos.")

    return result


def parameter_grid(full: bool) -> list[dict[str, float | str]]:
    quick = [
        {"C": 1.0, "gamma": "scale"},
        {"C": 3.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
        {"C": 30.0, "gamma": "scale"},
        {"C": 10.0, "gamma": 0.003},
        {"C": 10.0, "gamma": 0.01},
        {"C": 30.0, "gamma": 0.003},
        {"C": 30.0, "gamma": 0.01},
    ]
    if not full:
        return quick

    return [
        {"C": c, "gamma": gamma}
        for c in (0.5, 1.0, 3.0, 10.0, 30.0, 100.0)
        for gamma in ("scale", 0.001, 0.003, 0.01, 0.03)
    ]


def build_model(C: float, gamma: float | str) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    class_weight="balanced",
                    cache_size=2048,
                    probability=False,
                    random_state=RANDOM_STATE,
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

    for index, held_out in enumerate(subjects, start=1):
        test_mask = subject == held_out
        train_mask = ~test_mask

        fold_model = clone(model)
        fold_model.fit(X[train_mask], y[train_mask])
        fold_prediction = fold_model.predict(X[test_mask]).astype(str)
        predictions[test_mask] = fold_prediction

        fold_accuracy = accuracy_score(y[test_mask], fold_prediction)
        fold_f1 = f1_score(
            y[test_mask],
            fold_prediction,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        folds[held_out] = {
            "samples": int(test_mask.sum()),
            "accuracy": float(fold_accuracy),
            "macro_f1": float(fold_f1),
        }
        print(
            f"    [{index:02d}/{len(subjects):02d}] "
            f"{held_out:<15} accuracy={fold_accuracy:.4f} "
            f"macro_f1={fold_f1:.4f}"
        )

    predictions = predictions.astype(str)
    elapsed = time.perf_counter() - started

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
        "confusion_matrix": confusion_matrix(
            y, predictions, labels=labels
        ).tolist(),
        "folds": folds,
        "seconds": float(elapsed),
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

    candidates = parameter_grid(args.full)

    print("=" * 76)
    print("GESTUREBOT — BÚSQUEDA SVM RBF CON LOSO")
    print("=" * 76)
    print(f"Muestras: {len(y)} | Variables: {X.shape[1]} | Personas: {len(subjects)}")
    print(f"Configuraciones: {len(candidates)}")
    print()

    results: list[dict[str, Any]] = []

    for number, params in enumerate(candidates, start=1):
        print(
            f"[CONFIGURACIÓN {number:02d}/{len(candidates):02d}] "
            f"C={params['C']} gamma={params['gamma']}"
        )
        model = build_model(C=float(params["C"]), gamma=params["gamma"])
        metrics = evaluate_loso(model, X, y, subject, subjects, labels)

        result = {
            "C": float(params["C"]),
            "gamma": params["gamma"],
            **metrics,
        }
        results.append(result)

        print(
            f"  RESULTADO: accuracy={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"tiempo={metrics['seconds']:.1f}s\n"
        )

    best = max(
        results,
        key=lambda item: (item["macro_f1"], item["accuracy"]),
    )

    print("=" * 76)
    print("MEJOR CONFIGURACIÓN")
    print("=" * 76)
    print(f"C={best['C']} | gamma={best['gamma']}")
    print(f"Accuracy LOSO: {best['accuracy']:.4f}")
    print(f"Macro F1 LOSO: {best['macro_f1']:.4f}")
    print("Entrenando el modelo final con todas las personas...")

    final_model = build_model(C=best["C"], gamma=best["gamma"])
    final_model.fit(X, y)

    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "gesture_model_svm.joblib"
    joblib.dump(
        {
            "model": final_model,
            "model_name": "svm_rbf",
            "class_names": np.asarray(labels, dtype=str),
            "feature_names": feature_names,
            "number_of_features": int(X.shape[1]),
            "normalization_version": "wrist_origin_palm_scale_v1",
            "validation_strategy": "leave_one_subject_out",
            "trained_subjects": np.asarray(subjects, dtype=str),
            "parameters": {"C": best["C"], "gamma": best["gamma"]},
        },
        model_path,
    )

    report_path = reports_dir / "svm_loso_search.json"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
