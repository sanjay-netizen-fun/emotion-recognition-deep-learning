# 🎭 Emotion Recognition Using Deep Learning

A real-time facial emotion recognition system built using **Python, OpenCV, TensorFlow, and Keras**. The application uses a deep learning model to detect human faces through a webcam and classify their facial expressions into different emotion categories.

## 📌 Overview

This project implements a real-time facial emotion recognition system using computer vision and deep learning.

The system captures live video from a webcam, detects faces using OpenCV, processes the detected facial regions, and uses a trained deep learning model to predict the person's emotion.

The project was developed as an academic/deep learning project and has also been packaged as a **Windows desktop application** using PyInstaller and Inno Setup.

## ✨ Features

* 🎥 Real-time webcam-based emotion detection
* 🧑 Face detection using OpenCV
* 🧠 Deep learning-based emotion classification
* ⚡ Real-time prediction and display
* 🖥️ Windows desktop executable
* 📦 Windows installer support
* 📓 Jupyter Notebook containing the model training workflow
* 🤖 Trained Keras/TensorFlow model included

## 😊 Supported Emotions

The model is trained to recognize the following facial expressions:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## 🛠️ Technologies Used

| Technology           | Purpose                                       |
| -------------------- | --------------------------------------------- |
| **Python**           | Core programming language                     |
| **TensorFlow**       | Deep learning framework                       |
| **Keras**            | Neural network development and model handling |
| **OpenCV**           | Computer vision and webcam processing         |
| **NumPy**            | Numerical and array operations                |
| **Jupyter Notebook** | Model development and experimentation         |
| **PyInstaller**      | Windows executable packaging                  |
| **Inno Setup**       | Windows installer creation                    |

## 📂 Project Structure

```text
emotion-recognition-deep-learning/
│
├── realtimedetection.py          # Real-time emotion detection application
├── trainmodel.ipynb              # Model training notebook
├── emotiondetector.keras         # Trained deep learning model
├── emotiondetector.json          # Model configuration
├── face_recog.ico                # Application icon
├── Emotion Recognition.spec      # PyInstaller configuration
├── setup_fc_recog.iss            # Inno Setup installer script
├── requires.txt                  # Python dependencies
├── .gitignore                    # Git ignore configuration
└── README.md                     # Project documentation
```

> **Note:** The `images/`, `dist/`, and `installer/` folders are intentionally excluded from the Git repository. The dataset and generated Windows build files are kept locally and are not required in the source repository.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/sanjay-netizen-fun/emotion-recognition-deep-learning.git
```

### 2. Navigate to the project directory

```bash
cd emotion-recognition-deep-learning
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requires.txt
```

### 5. Run the application

```bash
python realtimedetection.py
```

Make sure your computer has a working webcam connected.

## 🖥️ Windows Application

The project has been successfully packaged into a standalone Windows application using **PyInstaller**.

The application can be built into:

```text
dist/
└── Emotion Recognition/
    └── Emotion Recognition.exe
```

A Windows installer was also created using **Inno Setup**.

The generated installer is intentionally excluded from the Git repository because of its large file size.

## 🧠 Model

The project uses a trained deep learning model stored as:

```text
emotiondetector.keras
```

The model receives facial image data and predicts the corresponding emotion class.

The model development and training process can be explored in:

```text
trainmodel.ipynb
```

## 🔄 How It Works

```text
Webcam
   ↓
Video Frame Capture
   ↓
Face Detection
   ↓
Face Region Extraction
   ↓
Image Preprocessing
   ↓
Deep Learning Model
   ↓
Emotion Prediction
   ↓
Emotion Displayed on Screen
```

## 📊 Dataset

The model was trained using facial expression image data organized according to emotion classes.

The dataset contains images representing different human facial expressions and is used during the model training and evaluation process.

The dataset itself is **not included in this GitHub repository** to keep the repository lightweight.

## 🔐 Repository Management

Large generated files and datasets are excluded using `.gitignore`.

The following directories are intentionally ignored:

```text
images/
dist/
installer/
venv/
__pycache__/
.ipynb_checkpoints/
```

This keeps the GitHub repository focused on the source code, trained model, configuration files, and documentation.

## 🚀 Future Improvements

Possible future improvements include:

* Improved model accuracy
* Better face detection under different lighting conditions
* Support for multiple faces simultaneously
* Emotion confidence scores
* Improved graphical user interface
* Mobile application integration
* Model optimization for lower-end devices
* Deployment as a web application

## 🎓 Project Purpose

This project demonstrates the practical application of:

* Deep Learning
* Computer Vision
* Facial Expression Recognition
* Machine Learning
* Python Programming
* TensorFlow/Keras
* Real-time image processing

It can serve as a foundation for further development in **AI, computer vision, and intelligent software applications**.

## 👨‍💻 Author

**Sanjay**

Developed as a deep learning and computer vision project.

## 📜 License

This project is intended for educational and academic purposes. If you plan to reuse or distribute the project, please review and comply with the licenses of the datasets, libraries, and other third-party components used.
