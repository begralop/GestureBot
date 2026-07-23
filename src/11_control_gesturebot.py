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
        self.hand_var = tk.StringVar(value="--")
        self.command_var = tk.StringVar(value="STOP")
        self.action_var = tk.StringVar(value="Robot detenido")
        self.connection_var = tk.StringVar(value="Modo simulación")
        self.performance_var = tk.StringVar(value="Inferencia: -- | FPS: --")

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
        self.root.geometry("1120x720")
        self.root.minsize(950, 650)

        title = tk.Label(
            self.root,
            text="Control de GestureBot",
            font=("Arial", 20, "bold"),
        )
        title.pack(pady=(14, 8))

        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=16, pady=8)

        left = tk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.video_label = tk.Label(
            left,
            text="Iniciando cámara...",
            bg="black",
            fg="white",
        )
        self.video_label.pack(fill="both", expand=True)

        right = tk.Frame(body, width=350, bd=1, relief="solid")
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(
            right,
            text="Predicción del modelo",
            font=("Arial", 15, "bold"),
        ).pack(pady=(18, 10))

        self._status_row(right, "Mano", self.hand_var)
        self._status_row(right, "Predicción", self.prediction_var)
        self._status_row(right, "Confianza", self.confidence_var)
        self._status_row(right, "Estabilidad", self.stability_var)
        self._status_row(right, "Comando", self.command_var)
        self._status_row(right, "Acción", self.action_var, wraplength=210)
        self._status_row(right, "Rendimiento", self.performance_var, wraplength=210)

        tk.Frame(right, height=1, bg="#cccccc").pack(
            fill="x", padx=18, pady=14
        )

        tk.Label(
            right,
            text="Comunicación",
            font=("Arial", 14, "bold"),
        ).pack(pady=(0, 8))

        tk.Checkbutton(
            right,
            text="Modo simulación (sin ESP32)",
            variable=self.simulation_var,
            command=self._on_mode_change,
            font=("Arial", 11),
        ).pack(anchor="w", padx=20, pady=4)

        ip_frame = tk.Frame(right)
        ip_frame.pack(fill="x", padx=20, pady=6)
        tk.Label(ip_frame, text="IP ESP32:").pack(side="left")
        tk.Entry(ip_frame, textvariable=self.ip_var, width=16).pack(
            side="right"
        )

        tk.Label(
            right,
            textvariable=self.connection_var,
            font=("Arial", 11, "bold"),
            wraplength=300,
            justify="left",
        ).pack(fill="x", padx=20, pady=8)

        tk.Button(
            right,
            text="PARADA DE EMERGENCIA",
            command=lambda: self.dispatch_command("STOP", force=True),
            bg="#cc3333",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
        ).pack(fill="x", padx=20, pady=(12, 8))

        tk.Label(
            right,
            text=(
                "Teclas de prueba: ← FORWARD · → BACKWARD · "
                "↓ STOP · ↑ GRIPPER"
            ),
            font=("Arial", 9),
            wraplength=300,
            justify="left",
        ).pack(fill="x", padx=20, pady=(8, 16))

    @staticmethod
    def _status_row(
        parent: tk.Widget,
        name: str,
        variable: tk.StringVar,
        wraplength: int = 220,
    ) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", padx=20, pady=5)
        tk.Label(
            row,
            text=f"{name}:",
            width=12,
            anchor="w",
            font=("Arial", 11, "bold"),
        ).pack(side="left", anchor="n")
        tk.Label(
            row,
            textvariable=variable,
            anchor="w",
            justify="left",
            wraplength=wraplength,
            font=("Arial", 11),
        ).pack(side="left", fill="x", expand=True)

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

    def _on_mode_change(self) -> None:
        if self.simulation_var.get():
            self.connection_var.set("Modo simulación: no se usa la red")
        else:
            self.connection_var.set(
                f"ESP32 activada: http://{self.ip_var.get().strip()}/"
            )

        # Al cambiar de modo, vuelve a aplicar el comando actual.
        self.last_dispatched_command = None
        self.dispatch_command(self.stable_command, force=True)

    def update_video(self, frame: np.ndarray) -> None:
        available_width = max(self.video_label.winfo_width(), 640)
        available_height = max(self.video_label.winfo_height(), 480)

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
                f"Modo simulación — comando generado: /{endpoint}"
            )
            return

        ip = self.ip_var.get().strip()
        if not ip:
            self.connection_var.set("No se ha indicado la IP de la ESP32")
            return

        self.connection_var.set(f"Enviando: http://{ip}/{endpoint}")

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
                        f"ESP32 confirmó el comando: {endpoint}"
                    ),
                )
            except (URLError, TimeoutError, OSError) as error:
                error_name = type(error).__name__
                self.root.after(
                    0,
                    lambda name=error_name: self.connection_var.set(
                        f"ESP32 no disponible ({name}). "
                        f"Comando simulado: {endpoint}"
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
            self.connection_var.set("No se pudo leer la cámara")
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
                self.connection_var.set(f"Frame descartado: {error}")
        else:
            self.probability_history.clear()
            self.candidate_label = "STOP"
            self.candidate_count = 0
            self.stable_command = "STOP"

        # Actualización de interfaz.
        self.hand_var.set(handedness if hand_detected else "No detectada")
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

        # Información directamente sobre la imagen.
        cv2.rectangle(frame, (0, 0), (500, 150), (18, 18, 18), -1)
        overlay_lines = [
            f"Prediccion: {raw_label}",
            f"Confianza: {confidence * 100:.1f} %",
            (
                "Estabilidad: "
                f"{min(self.candidate_count, self.args.confirm_frames)}/"
                f"{self.args.confirm_frames}"
            ),
            f"Comando: {self.stable_command}",
        ]
        for index, text in enumerate(overlay_lines):
            cv2.putText(
                frame,
                text,
                (15, 30 + index * 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
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
