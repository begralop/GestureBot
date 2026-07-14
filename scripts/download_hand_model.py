"""Descarga el modelo oficial Hand Landmarker de MediaPipe."""
from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "models" / "hand_landmarker.task"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"El modelo ya existe: {destination}")
        return
    print("Descargando modelo oficial de MediaPipe...")
    urlretrieve(MODEL_URL, destination)
    print(f"Modelo guardado en: {destination}")


if __name__ == "__main__":
    main()
