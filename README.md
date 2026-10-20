# 🎭 Emotion Recognition Using Deep Learning

A real-time facial emotion recognition system built with **Python, OpenCV, TensorFlow/Keras** — with a desktop app, a browser demo, a reproducible evaluation pipeline and an automated test suite.

The model detects a face in a webcam frame, crops it, converts it to a 48×48 grayscale image and classifies the expression into one of seven emotions.

## ✨ What's in the box

| Component | Description |
| --- | --- |
| **Desktop app** | `realtimedetection.py` — CustomTkinter UI with webcam preview, live emotion, confidence and FPS |
| **Web demo** | `webapp/` — Flask API + zero-build browser UI (webcam streaming or image upload, face boxes, score bars) |
| **Evaluation** | `evaluate.py` — accuracy, per-class precision/recall/F1, top-2 accuracy, confusion matrix, JSON/CSV/Markdown reports |
| **Training** | `train.py` — reproduces the shipped CNN from a folder-per-class dataset |
| **Core package** | `emotion_recognition/` — shared preprocessing, labels, detection, model wrapper and metrics |
| **Tests** | `tests/` — 98 pytest tests (fast unit tests + opt-in model/dataset integration tests) |
| **Model** | `emotiondetector.keras` (8.46M params) and `emotiondetector.json` architecture |

## 😊 Supported emotions

`Angry` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad` · `Surprise`

The class order comes from `sklearn.preprocessing.LabelEncoder` (alphabetical) applied to the training folder names. It is defined once in [`emotion_recognition/labels.py`](emotion_recognition/labels.py) and shared by every entry point, so the model's output index can never drift from the displayed label.

## 🛠️ Technologies

| Technology | Purpose |
| --- | --- |
| **Python 3.10+** | Core language |
| **TensorFlow / Keras** | Model training and inference |
| **OpenCV** | Haar-cascade face detection, image preprocessing, JPEG encoding |
| **NumPy** | Array operations and metrics |
| **Flask** | Web demo API |
| **CustomTkinter + Pillow** | Desktop GUI |
| **pytest** | Automated tests |
| **PyInstaller / Inno Setup** | Windows executable and installer |

## 🚀 Quick start

```bash
git clone https://github.com/sanjay-netizen-fun/emotion-recognition-deep-learning.git
cd emotion-recognition-deep-learning

python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requires.txt
```

### Option A — web demo (recommended for a quick look)

```bash
pip install -r requirements-web.txt
python webapp/app.py
```

Open <http://localhost:8000>, click **Start camera** (or drop an image onto the preview). The server binds to `0.0.0.0:8000`; use `--port` / `--host` to change it.

> The bundled Flask server is for local demos. For anything public, run it behind a production WSGI server (e.g. `gunicorn -w 1 'webapp.app:create_app()'`), noting that one worker per model copy is the practical limit without a shared inference service.

### Option B — desktop app

```bash
python realtimedetection.py
```

Requires a working webcam. On Windows the app opens the camera with DirectShow; other platforms fall back to the default backend.

### Option C — evaluate the model

```bash
python evaluate.py --data images --split validation --markdown docs/evaluation.md
```

### Option D — train from scratch

```bash
python train.py --data images --epochs 40 --output emotiondetector.keras
```

## 📊 Evaluation

The shipped model was evaluated on the **FER-2013 validation split** (3,589 images, never used for training):

| Metric | Value |
| --- | --- |
| Accuracy | **63.64 %** |
| Balanced accuracy | 64.46 % |
| Macro F1 | 0.617 |
| Weighted F1 | 0.607 |
| Top-2 accuracy | 80.61 % |
| Throughput | ~450 images/s (CPU, batching) |

Per-class results, the full confusion matrix and the most confused pairs are in **[docs/evaluation.md](docs/evaluation.md)**.

Key observations:

* **Happy** is by far the easiest class (F1 0.84); **Neutral** is the hardest (F1 0.24) and is mostly confused with Happy and Sad — a well-known FER-2013 characteristic.
* **Fear** has high precision but low recall (0.38): the model under-predicts it, absorbing many Fear faces into Sad.
* These numbers are typical for a small CNN trained on FER-2013 without augmentation or face alignment.

Regenerate the report at any time:

```bash
python evaluate.py --data images --split validation \
    --json runs/eval.json --csv runs/predictions.csv --markdown docs/evaluation.md
```

Useful flags: `--detect` (run Haar detection on each image first), `--no-fallback`, `--limit N`, `--shuffle`, `--min-accuracy 0.6` (exit code 1 below threshold, convenient in CI).

## 🌐 Web API

`webapp/app.py` exposes a small JSON API:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Version, model path, classes, detection availability, uptime |
| `POST` | `/api/warmup` | Load the model before the first frame (idempotent) |
| `GET` | `/api/classes` | Supported emotion labels |
| `POST` | `/api/analyze` | Run inference on an image |

`/api/analyze` accepts a multipart field `image` **or** JSON `{"image": "data:image/jpeg;base64,…"}`. Optional fields:

* `detect` (default `true`) — run Haar face detection; when `false` the full frame is classified
* `max_faces` (1–10, default `5`)
* `annotate` (default `false`) — include a base64 JPEG with boxes and labels

```bash
curl -s -X POST http://localhost:8000/api/analyze \
  -F image=@portrait.jpg -F detect=1 -F annotate=1 | python -m json.tool
```

```json
{
  "width": 928,
  "height": 1136,
  "count": 1,
  "fallback": false,
  "elapsed_ms": 174.2,
  "faces": [
    {
      "box": [229, 202, 466, 466],
      "emotion": "Happy",
      "confidence": 0.9951,
      "percent": 99.51,
      "scores": { "Angry": 0.0, "Happy": 0.9951, "Neutral": 0.0049, "...": 0.0 }
    }
  ]
}
```

Uploads are capped at 8 MB and processed in memory; nothing is written to disk.

## 🧠 Model details

* **Input:** 48×48 grayscale, scaled to `[0, 1]` — exactly the training pipeline
  (`keras.utils.load_img(..., color_mode="grayscale")` → `/255`).
* **Architecture:** 4 × Conv2D (128/256/512/512, 3×3, valid padding) each followed by max pooling and dropout, then `Flatten → Dense(512) → Dense(256) → Dense(7, softmax)`.
* **Parameters:** 4.23M trainable (8.46M including optimizer state).
* **Training:** Adam, categorical cross-entropy, batch size 128, early stopping on `val_loss` (patience 5).

Preprocessing lives in `emotion_recognition/preprocessing.py` and is **idempotent**: passing an already-normalized array returns the same values, which prevents the classic "divide by 255 twice" bug in batch pipelines.

## 📂 Project structure

```text
emotion-recognition-deep-learning/
├── realtimedetection.py        # Desktop app (CustomTkinter + webcam)
├── evaluate.py                 # Evaluation CLI (JSON/CSV/Markdown reports)
├── train.py                    # Training CLI
├── emotion_recognition/        # Shared core package
│   ├── labels.py               # Canonical class order
│   ├── preprocessing.py        # 48×48 grayscale + normalization
│   ├── detection.py            # Haar cascade wrapper
│   ├── model.py                # Model loading, batching, frame analysis
│   ├── metrics.py              # Confusion matrix, precision/recall/F1
│   ├── dataset.py              # Dataset discovery (split or flat layout)
│   └── training.py             # CNN definition + dataset loading
├── webapp/
│   ├── app.py                  # Flask app + JSON API
│   ├── render.py               # Server-side box annotation
│   └── static/                 # index.html, styles.css, app.js, favicon
├── tests/                      # pytest suite
├── docs/evaluation.md          # Generated evaluation report
├── emotiondetector.keras       # Trained model
├── emotiondetector.json        # Model architecture (Keras JSON)
├── Emotion Recognition.spec    # PyInstaller configuration
├── setup_fc_recog.iss          # Inno Setup installer script
├── requires.txt                # Runtime dependencies
├── requirements-web.txt        # Web demo dependencies
├── requirements-dev.txt        # Notebook, training and test extras
└── README.md
```

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest              # fast unit tests
pytest -m slow      # real model + dataset integration tests
```

The fast suite covers preprocessing (including the idempotency guard), metrics, dataset discovery, detection helpers, the model wrapper, annotation rendering, the evaluation CLI and the full Flask API — no TensorFlow required. The `slow` tests load the real model and skip automatically when `images/` is absent.

## 📊 Dataset

Training and evaluation use the FER-2013 image layout:

```text
images/
├── train/
│   ├── Angry/  Disgust/  Fear/  Happy/  Neutral/  Sad/  Surprise/
└── validation/
    └── Angry/  Disgust/  Fear/  Happy/  Neutral/  Sad/  Surprise/
```

Class folders are matched case-insensitively, and a flat `<class>/*.jpg` layout also works. The dataset is **not** included in this repository (see `.gitignore`); download FER-2013 from Kaggle or use any folder-per-class dataset.

## 🖥️ Windows executable

The desktop app can be packaged with PyInstaller:

```bash
pyinstaller "Emotion Recognition.spec"
```

Output: `dist/Emotion Recognition/Emotion Recognition.exe`. An Inno Setup script (`setup_fc_recog.iss`) builds the installer into `installer/`. Both `dist/` and `installer/` are ignored by Git.

## 🔐 Repository management

`images/`, `runs/`, `dist/`, `installer/`, `venv/`, caches and test artefacts are excluded via `.gitignore`. Generated evaluation reports (`docs/evaluation.md`) are small and committed on purpose.

## 🚀 Future improvements

* Data augmentation, face alignment and test-time augmentation to lift Neutral/Fear recall
* Temporal smoothing (majority vote over recent frames) to stabilise live predictions
* Multi-face tracking with stable identities
* ONNX / TFLite export for edge and mobile deployment
* A production inference service with request queuing and a shared model worker
* Optional GPU inference path and batched webcam requests

## 🎓 Project purpose

This project demonstrates practical deep learning and computer vision: a CNN trained from scratch, an OpenCV pipeline, desktop and web interfaces, reproducible evaluation, and an automated test suite that pins the preprocessing contract between training and inference.

## 👨‍💻 Author

**Sanjay** — built as a deep learning and computer vision project.

## 📜 License

Intended for educational and academic use. Review and comply with the licenses of the FER-2013 dataset, OpenCV, TensorFlow/Keras and the other third-party components before redistributing.
