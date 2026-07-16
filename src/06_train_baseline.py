"""Entrena modelos baseline para GestureBot.

Con una sola persona/sesión, la evaluación es provisional porque el split
aleatorio puede sobreestimar el rendimiento. Cuando haya varias sesiones o
personas, se usa un split por grupos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
CLASS_ORDER = ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara modelos baseline y guarda el mejor.")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/gesture_dataset.npz"))
    parser.add_argument("--test-size", type=float, default=0.25)
    return parser.parse_args()


def resolve(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset: {path}")
    data = np.load(path, allow_pickle=False)
    required = {"X", "y", "subject", "session", "feature_names"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Faltan arrays en el NPZ: {sorted(missing)}")
    result = {name: np.asarray(data[name]) for name in required}
    result["X"] = result["X"].astype(np.float32)
    if result["X"].ndim != 2 or not np.all(np.isfinite(result["X"])):
        raise ValueError("X no tiene una forma válida o contiene NaN/inf.")
    return result


def split_indices(y: np.ndarray, subject: np.ndarray, session: np.ndarray, test_size: float):
    groups = np.asarray([f"{s}::{sess}" for s, sess in zip(subject, session)], dtype=str)
    unique_groups = np.unique(groups)
    metadata: dict[str, Any] = {
        "unique_subjects": sorted(np.unique(subject).astype(str).tolist()),
        "unique_groups": sorted(unique_groups.tolist()),
    }
    if len(unique_groups) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
        strategy = "group_split_by_subject_session"
    else:
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        strategy = "stratified_random_split_provisional"
        metadata["warning"] = (
            "Solo existe una persona/sesión; la evaluación puede ser demasiado optimista."
        )
    metadata["train_groups"] = sorted(np.unique(groups[train_idx]).tolist())
    metadata["test_groups"] = sorted(np.unique(groups[test_idx]).tolist())
    return train_idx, test_idx, strategy, metadata


def build_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(96, 48),
                batch_size=64,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
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
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_comparison(results: dict[str, dict[str, float]], path: Path) -> None:
    names = list(results)
    accuracy_values = [results[name]["accuracy"] for name in names]
    f1_values = [results[name]["macro_f1"] for name in names]
    positions = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - width / 2, accuracy_values, width, label="Accuracy")
    ax.bar(positions + width / 2, f1_values, width, label="Macro F1")
    ax.set_ylabel("Puntuación")
    ax.set_title("Comparación de modelos baseline")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> int:
    args = parse_args()
    if not 0.05 <= args.test_size <= 0.5:
        print("--test-size debe estar entre 0.05 y 0.5.", file=sys.stderr)
        return 2

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = resolve(project_root, args.dataset)
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = load_dataset(dataset_path)
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    X = data["X"]
    y = data["y"].astype(str)
    subject = data["subject"].astype(str)
    session = data["session"].astype(str)
    feature_names = data["feature_names"].astype(str)

    train_idx, test_idx, strategy, split_meta = split_indices(y, subject, session, args.test_size)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    labels = [label for label in CLASS_ORDER if label in set(y.tolist())]

    print("=" * 72)
    print("GESTUREBOT — ENTRENAMIENTO BASELINE")
    print("=" * 72)
    print(f"Muestras: {len(y)} | Features: {X.shape[1]}")
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")
    print(f"Estrategia: {strategy}")
    if "warning" in split_meta:
        print("AVISO:", split_meta["warning"])
    print()

    trained = {}
    results: dict[str, dict[str, Any]] = {}

    for name, model in build_models().items():
        print(f"Entrenando {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        infer_start = time.perf_counter()
        pred = model.predict(X_test)
        infer_seconds = time.perf_counter() - infer_start

        accuracy = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, labels=labels, average="macro", zero_division=0)
        report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, pred, labels=labels)

        cm_path = figures_dir / f"confusion_matrix_{name}.png"
        save_confusion_matrix(matrix, labels, cm_path, f"Matriz de confusión — {name}")

        trained[name] = model
        results[name] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "training_seconds": float(train_seconds),
            "mean_inference_ms_per_sample": float(infer_seconds / max(len(X_test), 1) * 1000),
            "classification_report": report,
            "confusion_matrix": matrix,
            "confusion_matrix_path": str(cm_path.relative_to(project_root)),
        }
        print(f"  accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f}")

    best_name = max(results, key=lambda n: (results[n]["macro_f1"], results[n]["accuracy"]))
    model_path = models_dir / "gesture_model.joblib"
    artifact = {
        "model": trained[best_name],
        "class_names": np.asarray(labels, dtype=str),
        "feature_names": feature_names,
        "number_of_features": int(X.shape[1]),
        "normalization_version": "wrist_origin_palm_scale_v1",
        "split_strategy": strategy,
    }
    joblib.dump(artifact, model_path)

    comparison_path = figures_dir / "model_comparison.png"
    save_comparison(results, comparison_path)

    metadata = {
        "best_model": best_name,
        "model_path": str(model_path.relative_to(project_root)),
        "dataset_path": str(dataset_path.relative_to(project_root)),
        "number_of_samples": int(len(y)),
        "number_of_features": int(X.shape[1]),
        "classes": labels,
        "split_strategy": strategy,
        "split_metadata": split_meta,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "random_state": RANDOM_STATE,
        "metrics": {
            name: {
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "training_seconds": result["training_seconds"],
                "mean_inference_ms_per_sample": result["mean_inference_ms_per_sample"],
            }
            for name, result in results.items()
        },
        "warning": (
            "Resultados provisionales: no usar como métrica final hasta evaluar con personas o sesiones no vistas."
            if strategy == "stratified_random_split_provisional" else None
        ),
    }

    with (models_dir / "training_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(json_ready(metadata), file, indent=2, ensure_ascii=False)

    with (reports_dir / "baseline_results.json").open("w", encoding="utf-8") as file:
        json.dump(json_ready({"metadata": metadata, "models": results}), file, indent=2, ensure_ascii=False)

    print()
    print("-" * 72)
    print(f"Mejor modelo provisional: {best_name}")
    print(f"Accuracy: {results[best_name]['accuracy']:.4f}")
    print(f"Macro F1: {results[best_name]['macro_f1']:.4f}")
    print(f"Modelo guardado: {model_path.relative_to(project_root)}")
    print(f"Gráfica: {comparison_path.relative_to(project_root)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
