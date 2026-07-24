"""Control integrado de GestureBot: modelo IA + simulación/ESP32.

Este programa sustituye el control manual de prueba por el modelo real:

Webcam -> MediaPipe -> 128 features -> ensemble -> filtro temporal
       -> procesar_gesto() -> simulación o petición HTTP a la ESP32

Colocación recomendada dentro del proyecto:
    GestureBot/
    ├── models/
    │   ├── hand_landmarker.task
    │   └── gesture_model_ensemble.joblib
    └── src/
        └── 11_control_gesturebot.py

Ejecución en simulación (por defecto):
    python src/11_control_gesturebot.py

Ejecución intentando conectar con la ESP32:
    python src/11_control_gesturebot.py --live-esp

Dependencias:
    pip install opencv-python mediapipe numpy joblib scikit-learn pillow
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import messagebox
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

try:
    import cv2
    import joblib
    import mediapipe as mp
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as error:
    missing = getattr(error, "name", "una dependencia")
    raise SystemExit(
        f"Falta {missing}. Instala las dependencias con:\n"
        "pip install opencv-python mediapipe numpy joblib scikit-learn pillow"
    ) from error


CLASS_ORDER = ("FORWARD", "STOP", "BACKWARD", "GRIPPER", "OTHER")
PALM_MCP_INDICES = (5, 9, 13, 17)
EPSILON = 1e-8

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

COMMAND_CONFIG = {
    "FORWARD": {
        "endpoint": "avanzar",
        "gesture": "Palma / FORWARD",
        "action": "Avanzando",
    },
    "BACKWARD": {
        "endpoint": "retroceder",
        "gesture": "Dorso / BACKWARD",
        "action": "Retrocediendo",
    },
    "STOP": {
        "endpoint": "parar",
        "gesture": "Puño / STOP",
        "action": "Robot detenido",
    },
    "GRIPPER": {
        "endpoint": "pinza",
        "gesture": "Gesto OK / GRIPPER",
        "action": "Abriendo o cerrando la pinza",
    },
    "OTHER": {
        "endpoint": "parar",
        "gesture": "OTHER",
        "action": "Robot detenido por seguridad",
    },
}


UI = {
    "bg": "#0b1220",
    "panel": "#121a2b",
    "card": "#182338",
    "card_alt": "#1d2942",
    "border": "#2b3a58",
    "text": "#e5eefb",
    "muted": "#9fb0cf",
    "accent": "#5eead4",
    "accent_2": "#60a5fa",
    "good": "#22c55e",
    "warn": "#f59e0b",
    "danger": "#ef4444",
    "video_bg": "#030712",
}

FONT = {
    "title": ("Segoe UI", 24, "bold"),
    "subtitle": ("Segoe UI", 10),
    "section": ("Segoe UI", 12, "bold"),
    "label": ("Segoe UI", 10, "bold"),
    "value": ("Segoe UI", 10),
    "value_big": ("Segoe UI", 11, "bold"),
    "button": ("Segoe UI", 10, "bold"),
    "badge": ("Segoe UI", 9, "bold"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GestureBot: modelo en tiempo real y control simulado/ESP32."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--hand-model", type=Path, default=None)
    parser.add_argument("--classifier", type=Path, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--confirm-frames", type=int, default=4)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--esp-ip", default="192.168.4.1")
    parser.add_argument(
        "--live-esp",
        action="store_true",
        help="Intenta enviar los comandos a la ESP32. Sin esta opción simula.",
    )
    return parser.parse_args()


def find_project_root() -> Path:
    """Localiza la raíz tanto si el archivo está en src/ como en la raíz."""
    script_dir = Path(__file__).resolve().parent
    candidates = (script_dir.parent, script_dir)

    for candidate in candidates:
        if (candidate / "models").is_dir():
            return candidate
    return script_dir.parent


def corrected_handedness(category_name: str | None) -> str:
    """Corrige la lateralidad porque la imagen se muestra en espejo."""
    if category_name == "Left":
        return "RIGHT"
    if category_name == "Right":
        return "LEFT"
    return "UNKNOWN"


def landmarks_to_array(landmarks: list) -> np.ndarray:
    array = np.asarray(
        [[float(point.x), float(point.y), float(point.z)] for point in landmarks],
        dtype=np.float32,
    )
    if array.shape != (21, 3):
        raise ValueError(f"Forma inesperada de landmarks: {array.shape}")
    return array


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Coloca la muñeca en el origen y escala por el tamaño de la palma."""
    centered = landmarks - landmarks[0]
    palm_vectors = centered[list(PALM_MCP_INDICES)]
    palm_distances = np.linalg.norm(palm_vectors, axis=1)
    scale = float(np.mean(palm_distances))

    if not math.isfinite(scale) or scale <= EPSILON:
        raise ValueError("No se puede calcular una escala válida.")

    return (centered / scale).astype(np.float32)


def handedness_features(handedness: str) -> np.ndarray:
    if handedness == "LEFT":
        return np.asarray([1.0, 0.0], dtype=np.float32)
    if handedness == "RIGHT":
        return np.asarray([0.0, 1.0], dtype=np.float32)
    raise ValueError(f"Lateralidad no válida: {handedness}")


def build_feature_vector(
    image_landmarks: list,
    world_landmarks: list,
    handedness: str,
) -> np.ndarray:
    image_array = landmarks_to_array(image_landmarks)
    world_array = landmarks_to_array(world_landmarks)

    image_normalized = normalize_landmarks(image_array)
    world_normalized = normalize_landmarks(world_array)

    vector = np.concatenate(
        (
            image_normalized.reshape(-1),
            world_normalized.reshape(-1),
            handedness_features(handedness),
        )
    ).astype(np.float32)

    if vector.shape != (128,):
        raise ValueError(
            f"Se esperaban 128 características y hay {vector.shape[0]}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("El vector contiene NaN o infinitos.")

    return vector


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    values -= np.max(values)
    exponentials = np.exp(values)
    denominator = np.sum(exponentials)
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError("No se pueden normalizar las puntuaciones del modelo.")
    return exponentials / denominator


def aligned_model_probabilities(
    model: Any,
    sample: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """Obtiene las probabilidades de un modelo en el orden común de clases."""
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(sample)[0], dtype=np.float64)
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(sample))
        if decision.ndim == 2:
            decision = decision[0]
        raw = softmax(decision)
    else:
        prediction = str(model.predict(sample)[0])
        raw = np.zeros(len(labels), dtype=np.float64)
        raw[labels.index(prediction)] = 1.0
        return raw

    model_classes = [str(value) for value in model.classes_]
    aligned = np.zeros(len(labels), dtype=np.float64)

    for target_index, label in enumerate(labels):
        if label not in model_classes:
            raise ValueError(f"El modelo no contiene la clase {label}.")
        aligned[target_index] = raw[model_classes.index(label)]

    return aligned


def predict_artifact(
    artifact: dict[str, Any],
    feature_vector: np.ndarray,
) -> tuple[str, float, np.ndarray, list[str]]:
    labels = [str(value) for value in artifact["class_names"]]
    sample = feature_vector.reshape(1, -1)

    if "models" in artifact:
        models = artifact["models"]
        weights = artifact.get("weights", {})
        combined = np.zeros(len(labels), dtype=np.float64)
        total_weight = 0.0

        for model_name, model in models.items():
            weight = float(weights.get(model_name, 1.0))
            if weight <= 0:
                continue
            combined += weight * aligned_model_probabilities(
                model, sample, labels
            )
            total_weight += weight

        if total_weight <= 0:
            raise ValueError("Los pesos del ensemble no son válidos.")
        probabilities = combined / total_weight

    elif "model" in artifact:
        probabilities = aligned_model_probabilities(
            artifact["model"], sample, labels
        )
    else:
        raise ValueError("El artefacto no contiene 'models' ni 'model'.")

    best_index = int(np.argmax(probabilities))
    return (
        labels[best_index],
        float(probabilities[best_index]),
        probabilities,
        labels,
    )


def draw_hand(frame: np.ndarray, landmarks: list) -> None:
    height, width = frame.shape[:2]
    points = [
        (
            int(np.clip(point.x, 0.0, 1.0) * width),
            int(np.clip(point.y, 0.0, 1.0) * height),
        )
        for point in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (40, 210, 255),
            2,
            cv2.LINE_AA,
        )

    for point in points:
        cv2.circle(frame, point, 4, (255, 80, 80), -1, cv2.LINE_AA)


class GestureBotApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.project_root = find_project_root()

        self.hand_model_path = args.hand_model or (
            self.project_root / "models" / "hand_landmarker.task"
        )
        self.classifier_path = args.classifier or (
            self.project_root / "models" / "gesture_model_ensemble.joblib"
        )

        self.artifact: dict[str, Any] | None = None
        self.capture: cv2.VideoCapture | None = None
        self.landmarker: Any = None

        self.probability_history: deque[np.ndarray] = deque(
            maxlen=args.smoothing_window
        )
        self.candidate_label = "STOP"
        self.candidate_count = 0
        self.stable_command = "STOP"
        self.last_dispatched_command: str | None = None

        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1
        self.last_frame_time = time.perf_counter()
        self.fps_history: deque[float] = deque(maxlen=30)
        self.closing = False
        self.photo_image: ImageTk.PhotoImage | None = None

        self.simulation_var = tk.BooleanVar(value=not args.live_esp)
        self.ip_var = tk.StringVar(value=args.esp_ip)
        self.prediction_var = tk.StringVar(value="--")
        self.confidence_var = tk.StringVar(value="--")
        self.stability_var = tk.StringVar(value=f"0/{args.confirm_frames}")
        self.hand_var = tk.StringVar(value="No hand detected")
        self.command_var = tk.StringVar(value="STOP")
        self.action_var = tk.StringVar(value="Robot stopped")
        self.connection_var = tk.StringVar(value="Simulation mode active")
        self.performance_var = tk.StringVar(value="Inference: -- | FPS: --")

        self._validate_arguments()
        self._load_resources()
        self._build_interface()
        self._bind_manual_controls()

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(10, self.process_frame)

    def _validate_arguments(self) -> None:
        if not 0.0 <= self.args.min_confidence <= 1.0:
            raise ValueError("--min-confidence debe estar entre 0 y 1.")
        if self.args.confirm_frames < 1:
            raise ValueError("--confirm-frames debe ser al menos 1.")
        if self.args.smoothing_window < 1:
            raise ValueError("--smoothing-window debe ser al menos 1.")

    def _load_resources(self) -> None:
        if not self.hand_model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo de MediaPipe:\n{self.hand_model_path}"
            )
        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"No existe el ensemble:\n{self.classifier_path}"
            )

        loaded = joblib.load(self.classifier_path)
        if not isinstance(loaded, dict):
            raise ValueError("El archivo joblib no contiene un artefacto válido.")
        self.artifact = loaded

        expected_features = int(self.artifact.get("number_of_features", 128))
        if expected_features != 128:
            raise ValueError(
                f"El modelo espera {expected_features} características, no 128."
            )

        self.capture = cv2.VideoCapture(self.args.camera)
        if not self.capture.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {self.args.camera}.")

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.hand_model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )

    def _build_interface(self) -> None:
        self.root.title("Control GestureBot — IA + ESP32")
        self.root.geometry("1320x840")
        self.root.minsize(1160, 760)
        self.root.configure(bg=UI["bg"])

        header = tk.Frame(self.root, bg=UI["bg"])
        header.pack(fill="x", padx=22, pady=(18, 8))

        title_col = tk.Frame(header, bg=UI["bg"])
        title_col.pack(side="left", fill="x", expand=True)
        tk.Label(
            title_col,
            text="GestureBot Control Center",
            bg=UI["bg"],
            fg=UI["text"],
            font=FONT["title"],
        ).pack(anchor="w")
        tk.Label(
            title_col,
            text=(
                "Webcam + MediaPipe + ensemble + temporal filter + "
                "simulation/ESP32 control"
            ),
            bg=UI["bg"],
            fg=UI["muted"],
            font=FONT["subtitle"],
        ).pack(anchor="w", pady=(4, 0))

        self.mode_badge = tk.Label(
            header,
            text="SIMULATION MODE",
            bg="#133127",
            fg="#8ef0b0",
            font=FONT["badge"],
            padx=14,
            pady=8,
        )
        self.mode_badge.pack(side="right", anchor="ne")

        body = tk.Frame(self.root, bg=UI["bg"])
        body.pack(fill="both", expand=True, padx=22, pady=(6, 18))

        left = tk.Frame(body, bg=UI["panel"], highlightbackground=UI["border"], highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))

        video_header = tk.Frame(left, bg=UI["panel"])
        video_header.pack(fill="x", padx=16, pady=(14, 8))
        tk.Label(
            video_header,
            text="Live camera",
            bg=UI["panel"],
            fg=UI["text"],
            font=FONT["section"],
        ).pack(anchor="w")
        tk.Label(
            video_header,
            text="Real-time preview with landmarks and command overlay",
            bg=UI["panel"],
            fg=UI["muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

        video_wrap = tk.Frame(left, bg=UI["video_bg"], highlightbackground=UI["border"], highlightthickness=1)
        video_wrap.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.video_label = tk.Label(
            video_wrap,
            text="Starting camera...",
            bg=UI["video_bg"],
            fg=UI["muted"],
            font=("Segoe UI", 12),
        )
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        right = tk.Frame(body, bg=UI["panel"], width=390, highlightbackground=UI["border"], highlightthickness=1)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        scroll = tk.Frame(right, bg=UI["panel"])
        scroll.pack(fill="both", expand=True, padx=14, pady=14)

        self._make_section_title(scroll, "Model status", "Live output of the recognition pipeline")
        metrics = tk.Frame(scroll, bg=UI["panel"])
        metrics.pack(fill="x", pady=(0, 10))
        self._status_row(metrics, "Hand", self.hand_var)
        self._status_row(metrics, "Prediction", self.prediction_var)
        self._status_row(metrics, "Confidence", self.confidence_var)
        self._status_row(metrics, "Stability", self.stability_var)
        self._status_row(metrics, "Command", self.command_var)
        self._status_row(metrics, "Action", self.action_var, wraplength=240)
        self._status_row(metrics, "Performance", self.performance_var, wraplength=240)

        self._make_section_title(scroll, "Communication", "Switch between simulation and the physical ESP32")
        comm_card = tk.Frame(scroll, bg=UI["card"], highlightbackground=UI["border"], highlightthickness=1)
        comm_card.pack(fill="x", pady=(0, 10))

        top_line = tk.Frame(comm_card, bg=UI["card"])
        top_line.pack(fill="x", padx=14, pady=(14, 8))
        tk.Checkbutton(
            top_line,
            text="Simulation mode (without ESP32)",
            variable=self.simulation_var,
            command=self._on_mode_change,
            bg=UI["card"],
            fg=UI["text"],
            activebackground=UI["card"],
            activeforeground=UI["text"],
            selectcolor=UI["card_alt"],
            font=("Segoe UI", 10),
            highlightthickness=0,
            bd=0,
        ).pack(anchor="w")

        ip_line = tk.Frame(comm_card, bg=UI["card"])
        ip_line.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(
            ip_line,
            text="ESP32 IP",
            bg=UI["card"],
            fg=UI["muted"],
            font=FONT["label"],
        ).pack(anchor="w")
        self.ip_entry = tk.Entry(
            ip_line,
            textvariable=self.ip_var,
            bg=UI["card_alt"],
            fg=UI["text"],
            insertbackground=UI["text"],
            relief="flat",
            font=("Consolas", 11),
            highlightthickness=1,
            highlightbackground=UI["border"],
            highlightcolor=UI["accent_2"],
        )
        self.ip_entry.pack(fill="x", pady=(6, 0), ipady=7)
        self.ip_entry.bind("<Return>", self._apply_ip)
        self.ip_entry.bind("<FocusOut>", self._apply_ip)

        btn_row = tk.Frame(comm_card, bg=UI["card"])
        btn_row.pack(fill="x", padx=14, pady=(8, 10))
        tk.Button(
            btn_row,
            text="TEST CONNECTION",
            command=self._test_connection,
            bg=UI["accent_2"],
            fg="#07111f",
            activebackground="#3b82f6",
            activeforeground="#07111f",
            relief="flat",
            font=FONT["button"],
            cursor="hand2",
            padx=10,
            pady=9,
        ).pack(side="left", fill="x", expand=True)

        tk.Label(
            comm_card,
            textvariable=self.connection_var,
            bg=UI["card"],
            fg=UI["text"],
            font=("Segoe UI", 9),
            wraplength=320,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 14))

        self._make_section_title(scroll, "Quick actions", "Emergency stop and keyboard shortcuts")
        action_card = tk.Frame(scroll, bg=UI["card"], highlightbackground=UI["border"], highlightthickness=1)
        action_card.pack(fill="x", pady=(0, 10))
        tk.Button(
            action_card,
            text="EMERGENCY STOP",
            command=lambda: self.dispatch_command("STOP", force=True),
            bg=UI["danger"],
            fg="white",
            activebackground="#dc2626",
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 12, "bold"),
            cursor="hand2",
            padx=10,
            pady=12,
        ).pack(fill="x", padx=14, pady=(14, 10))
        tk.Label(
            action_card,
            text=(
                "Test keys:\n"
                "← Forward   ·   → Backward\n"
                "↓ Stop      ·   ↑ Gripper"
            ),
            bg=UI["card"],
            fg=UI["muted"],
            font=("Segoe UI", 9),
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 14))

        footer = tk.Label(
            scroll,
            text="Tip: click on the video area or the window before using the arrow keys.",
            bg=UI["panel"],
            fg=UI["muted"],
            font=("Segoe UI", 8),
            justify="left",
        )
        footer.pack(anchor="w", pady=(4, 0))

        self._refresh_mode_badge()

    def _make_section_title(self, parent: tk.Widget, title: str, subtitle: str) -> None:
        frame = tk.Frame(parent, bg=UI["panel"])
        frame.pack(fill="x", pady=(0, 8))
        tk.Label(
            frame,
            text=title,
            bg=UI["panel"],
            fg=UI["text"],
            font=FONT["section"],
        ).pack(anchor="w")
        tk.Label(
            frame,
            text=subtitle,
            bg=UI["panel"],
            fg=UI["muted"],
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(1, 0))

    def _status_row(
        self,
        parent: tk.Widget,
        name: str,
        variable: tk.StringVar,
        wraplength: int = 220,
    ) -> None:
        card = tk.Frame(
            parent,
            bg=UI["card"],
            highlightbackground=UI["border"],
            highlightthickness=1,
        )
        card.pack(fill="x", pady=5)
        tk.Label(
            card,
            text=name.upper(),
            bg=UI["card"],
            fg=UI["muted"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=14, pady=(10, 2))
        tk.Label(
            card,
            textvariable=variable,
            bg=UI["card"],
            fg=UI["text"],
            anchor="w",
            justify="left",
            wraplength=wraplength,
            font=FONT["value_big"],
        ).pack(fill="x", padx=14, pady=(0, 10))

    def _bind_manual_controls(self) -> None:
        key_map = {
            "Left": "FORWARD",
            "Right": "BACKWARD",
            "Down": "STOP",
            "Up": "GRIPPER",
        }

        def on_key(event: tk.Event) -> None:
            # No interpreta las flechas como comandos mientras se edita la IP.
            if isinstance(event.widget, tk.Entry):
                return
            label = key_map.get(str(event.keysym))
            if label:
                self.dispatch_command(label, force=True)

        self.root.bind("<KeyPress>", on_key)
        self.root.focus_force()

    def _refresh_mode_badge(self) -> None:
        if self.simulation_var.get():
            self.mode_badge.config(
                text="SIMULATION MODE",
                bg="#133127",
                fg="#8ef0b0",
            )
        else:
            self.mode_badge.config(
                text="LIVE ESP32 MODE",
                bg="#14243f",
                fg="#9cc5ff",
            )

    def _apply_ip(self, _event: tk.Event | None = None) -> None:
        self.last_dispatched_command = None
        if self.simulation_var.get():
            self.connection_var.set(
                f"ESP32 IP saved: {self.ip_var.get().strip()} (simulation mode active)"
            )
        else:
            self.connection_var.set(
                f"ESP32 active: http://{self.ip_var.get().strip()}/"
            )
            self.dispatch_command(self.stable_command, force=True)

    def _test_connection(self) -> None:
        if self.simulation_var.get():
            self.connection_var.set(
                "Simulation mode is active. Disable it to test the real ESP32."
            )
            return
        self.last_dispatched_command = None
        self.dispatch_command("STOP", force=True)

    def _on_mode_change(self) -> None:
        self._refresh_mode_badge()
        if self.simulation_var.get():
            self.connection_var.set("Simulation mode active: network requests disabled")
        else:
            self.connection_var.set(
                f"ESP32 active: http://{self.ip_var.get().strip()}/"
            )

        self.last_dispatched_command = None
        self.dispatch_command(self.stable_command, force=True)

    def update_video(self, frame: np.ndarray) -> None:
        available_width = max(self.video_label.winfo_width(), 760)
        available_height = max(self.video_label.winfo_height(), 560)

        height, width = frame.shape[:2]
        scale = min(available_width / width, available_height / height)
        new_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        )

        resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.photo_image = ImageTk.PhotoImage(image=image)
        self.video_label.config(image=self.photo_image, text="")

    def dispatch_command(self, label: str, force: bool = False) -> None:
        """Equivalente integrado a procesar_gesto() del Control.py original."""
        label = label.upper().strip()
        if label not in COMMAND_CONFIG:
            return

        # GRIPPER se manda una sola vez al entrar en el gesto. Los demás
        # comandos tampoco se repiten hasta que el estado cambia.
        if not force and label == self.last_dispatched_command:
            return

        self.last_dispatched_command = label
        config = COMMAND_CONFIG[label]
        endpoint = str(config["endpoint"])

        self.command_var.set(label)
        self.action_var.set(str(config["action"]))

        if self.simulation_var.get():
            self.connection_var.set(
                f"Simulation mode — generated command: /{endpoint}"
            )
            return

        ip = self.ip_var.get().strip()
        if not ip:
            self.connection_var.set("No ESP32 IP has been provided")
            return

        self.connection_var.set(f"Sending: http://{ip}/{endpoint}")

        def request() -> None:
            try:
                with urlopen(
                    f"http://{ip}/{endpoint}",
                    timeout=0.5,
                ) as response:
                    response.read()
                self.root.after(
                    0,
                    lambda: self.connection_var.set(
                        f"ESP32 confirmed command: {endpoint}"
                    ),
                )
            except (URLError, TimeoutError, OSError) as error:
                error_name = type(error).__name__
                self.root.after(
                    0,
                    lambda name=error_name: self.connection_var.set(
                        f"ESP32 unavailable ({name}). Fallback info: {endpoint}"
                    ),
                )

        threading.Thread(target=request, daemon=True).start()

    def process_frame(self) -> None:
        if self.closing:
            return
        if self.capture is None or self.landmarker is None or self.artifact is None:
            return

        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.connection_var.set("Could not read from the camera")
            self.root.after(100, self.process_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_detected = bool(
            result.hand_landmarks and result.hand_world_landmarks
        )
        handedness = "--"
        raw_label = "--"
        confidence = 0.0
        inference_ms = 0.0

        if hand_detected:
            image_landmarks = result.hand_landmarks[0]
            world_landmarks = result.hand_world_landmarks[0]
            draw_hand(frame, image_landmarks)

            if result.handedness and result.handedness[0]:
                category = result.handedness[0][0]
                handedness = corrected_handedness(category.category_name)

            try:
                feature_vector = build_feature_vector(
                    image_landmarks,
                    world_landmarks,
                    handedness,
                )

                inference_start = time.perf_counter()
                _, _, probabilities, labels = predict_artifact(
                    self.artifact,
                    feature_vector,
                )
                inference_ms = (
                    time.perf_counter() - inference_start
                ) * 1000.0

                self.probability_history.append(probabilities)
                smoothed = np.mean(
                    np.stack(self.probability_history, axis=0),
                    axis=0,
                )
                best_index = int(np.argmax(smoothed))
                raw_label = labels[best_index]
                confidence = float(smoothed[best_index])

                accepted_label = (
                    raw_label
                    if confidence >= self.args.min_confidence
                    and raw_label != "OTHER"
                    else "STOP"
                )

                # STOP se ejecuta inmediatamente por seguridad.
                if accepted_label == "STOP":
                    self.stable_command = "STOP"
                    self.candidate_label = "STOP"
                    self.candidate_count = self.args.confirm_frames
                else:
                    if accepted_label == self.candidate_label:
                        self.candidate_count += 1
                    else:
                        self.candidate_label = accepted_label
                        self.candidate_count = 1

                    if self.candidate_count >= self.args.confirm_frames:
                        self.stable_command = self.candidate_label

            except ValueError as error:
                raw_label = "ERROR"
                confidence = 0.0
                self.stable_command = "STOP"
                self.candidate_label = "STOP"
                self.candidate_count = 0
                self.probability_history.clear()
                self.connection_var.set(f"Discarded frame: {error}")
        else:
            self.probability_history.clear()
            self.candidate_label = "STOP"
            self.candidate_count = 0
            self.stable_command = "STOP"

        # Actualización de interfaz.
        self.hand_var.set(handedness if hand_detected else "No hand detected")
        self.prediction_var.set(raw_label)
        self.confidence_var.set(
            f"{confidence * 100:.1f} %" if hand_detected else "--"
        )
        self.stability_var.set(
            f"{min(self.candidate_count, self.args.confirm_frames)}/"
            f"{self.args.confirm_frames}"
        )

        now = time.perf_counter()
        elapsed = now - self.last_frame_time
        self.last_frame_time = now
        if elapsed > 0:
            self.fps_history.append(1.0 / elapsed)
        mean_fps = float(np.mean(self.fps_history)) if self.fps_history else 0.0
        self.performance_var.set(
            f"Inferencia ensemble: {inference_ms:.2f} ms | "
            f"FPS: {mean_fps:.1f}"
        )

        # Esta es la llamada que conecta la salida estable del modelo con
        # la lógica que preparó tu compañero.
        self.dispatch_command(self.stable_command)

        # Información elegante sobre la imagen.
        panel = frame.copy()
        cv2.rectangle(panel, (16, 16), (420, 158), (11, 18, 32), -1)
        cv2.rectangle(panel, (16, 16), (420, 158), (43, 58, 88), 1)
        cv2.addWeighted(panel, 0.92, frame, 0.08, 0, frame)
        cv2.rectangle(frame, (16, 16), (420, 52), (34, 197, 94), -1)
        cv2.putText(
            frame,
            "LIVE PREDICTION",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (7, 18, 31),
            2,
            cv2.LINE_AA,
        )
        overlay_lines = [
            f"Gesture: {raw_label}",
            f"Confidence: {confidence * 100:.1f} %",
            (
                "Stability: "
                f"{min(self.candidate_count, self.args.confirm_frames)}/"
                f"{self.args.confirm_frames}"
            ),
            f"Command: {self.stable_command}",
        ]
        for index, text_line in enumerate(overlay_lines):
            cv2.putText(
                frame,
                text_line,
                (30, 78 + index * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (229, 238, 251),
                1,
                cv2.LINE_AA,
            )

        self.update_video(frame)
        self.root.after(1, self.process_frame)

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True

        # Fuerza STOP antes de cerrar cuando está activada la ESP32.
        if not self.simulation_var.get():
            self.dispatch_command("STOP", force=True)
            time.sleep(0.05)

        if self.capture is not None:
            self.capture.release()
        if self.landmarker is not None:
            self.landmarker.close()

        self.root.destroy()


def main() -> int:
    args = parse_args()
    root = tk.Tk()

    try:
        GestureBotApp(root, args)
    except Exception as error:
        root.withdraw()
        messagebox.showerror("No se pudo iniciar GestureBot", str(error))
        root.destroy()
        print(f"Error: {error}", file=sys.stderr)
        return 1

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
