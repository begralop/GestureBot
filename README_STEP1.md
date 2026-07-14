# GestureBot — Paso 1

Objetivo: comprobar webcam y visualizar los 21 landmarks de una mano. Todavía no se recopilan datos ni se clasifican gestos.

## 1. Crear y activar el entorno

```bash
cd /ruta/a/GestureBot_step1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Probar la webcam

```bash
python src/01_test_camera.py
```

## 3. Descargar el modelo oficial

```bash
python scripts/download_hand_model.py
```

## 4. Probar MediaPipe

```bash
python src/02_show_hand_landmarks.py
```

Debes ver 21 puntos, sus conexiones, mano izquierda/derecha y FPS. Cierra con `q` o `ESC`.

Si la cámara no abre, activa el permiso en **Ajustes del Sistema > Privacidad y seguridad > Cámara**. También puedes probar `--camera 1`.
