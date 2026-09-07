import os
import sys
import cv2
import time
import customtkinter as ctk
from PIL import Image, ImageTk
from keras.models import model_from_json
import numpy as np


def resource_path(filename):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


with open(resource_path("emotiondetector.json"), "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights(resource_path("emotiondetector.keras"))

haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion")
        self.root.geometry("1000x680")
        self.root.minsize(900, 620)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.webcam = None
        self.running = False
        self.last_time = time.time()
        self.fps = 0

        self.setup_ui()

    def setup_ui(self):
        self.root.configure(fg_color="#101010")

        top = ctk.CTkFrame(
            self.root,
            fg_color="transparent"
        )
        top.pack(fill="x", padx=35, pady=(25, 15))

        title = ctk.CTkLabel(
            top,
            text="EMOTION",
            font=("Arial", 22, "bold")
        )
        title.pack(side="left")

        subtitle = ctk.CTkLabel(
            top,
            text="REAL-TIME RECOGNITION",
            text_color="#888888",
            font=("Arial", 12)
        )
        subtitle.pack(side="left", padx=12, pady=(5, 0))

        self.status = ctk.CTkLabel(
            top,
            text="● OFFLINE",
            text_color="#777777",
            font=("Arial", 12, "bold")
        )
        self.status.pack(side="right")

        content = ctk.CTkFrame(
            self.root,
            fg_color="transparent"
        )
        content.pack(
            fill="both",
            expand=True,
            padx=35,
            pady=10
        )

        self.camera_frame = ctk.CTkFrame(
            content,
            fg_color="#181818",
            corner_radius=18
        )
        self.camera_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 15)
        )

        self.video_label = ctk.CTkLabel(
            self.camera_frame,
            text="Camera is off",
            text_color="#666666",
            font=("Arial", 18)
        )
        self.video_label.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10
        )

        info = ctk.CTkFrame(
            content,
            width=260,
            fg_color="#181818",
            corner_radius=18
        )
        info.pack(
            side="right",
            fill="y"
        )
        info.pack_propagate(False)

        ctk.CTkLabel(
            info,
            text="DETECTED",
            text_color="#777777",
            font=("Arial", 11, "bold")
        ).pack(pady=(45, 5))

        self.emotion_label = ctk.CTkLabel(
            info,
            text="—",
            font=("Arial", 34, "bold")
        )
        self.emotion_label.pack(pady=(0, 35))

        ctk.CTkLabel(
            info,
            text="CONFIDENCE",
            text_color="#777777",
            font=("Arial", 11, "bold")
        ).pack()

        self.confidence_label = ctk.CTkLabel(
            info,
            text="0%",
            font=("Arial", 24)
        )
        self.confidence_label.pack(pady=(4, 30))

        ctk.CTkLabel(
            info,
            text="FPS",
            text_color="#777777",
            font=("Arial", 11, "bold")
        ).pack()

        self.fps_label = ctk.CTkLabel(
            info,
            text="0",
            font=("Arial", 24)
        )
        self.fps_label.pack(pady=(4, 40))

        self.start_button = ctk.CTkButton(
            info,
            text="Start camera",
            height=42,
            corner_radius=10,
            command=self.start_camera
        )
        self.start_button.pack(
            padx=30,
            fill="x",
            pady=(10, 8)
        )

        self.stop_button = ctk.CTkButton(
            info,
            text="Stop",
            height=42,
            corner_radius=10,
            fg_color="#252525",
            hover_color="#303030",
            command=self.stop_camera,
            state="disabled"
        )
        self.stop_button.pack(
            padx=30,
            fill="x"
        )

        footer = ctk.CTkLabel(
            self.root,
            text="Deep Learning  •  OpenCV  •  Keras",
            text_color="#555555",
            font=("Arial", 11)
        )
        footer.pack(pady=(5, 20))

    def start_camera(self):
        if self.running:
            return

        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.webcam.isOpened():
            self.status.configure(
                text="● CAMERA ERROR",
                text_color="#cc5555"
            )
            return

        self.running = True
        self.last_time = time.time()

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        self.status.configure(
            text="● LIVE",
            text_color="#55bb77"
        )

        self.update_frame()

    def stop_camera(self):
        self.running = False

        if self.webcam is not None:
            self.webcam.release()
            self.webcam = None

        self.video_label.configure(
            image=None,
            text="Camera is off"
        )

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

        self.status.configure(
            text="● OFFLINE",
            text_color="#777777"
        )

        self.emotion_label.configure(text="—")
        self.confidence_label.configure(text="0%")
        self.fps_label.configure(text="0")

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.webcam.read()

        if not ret:
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        emotion = "No face"
        confidence = 0

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))

            prediction = model.predict(
                extract_features(face),
                verbose=0
            )

            index = prediction.argmax()
            emotion = labels[index]
            confidence = float(prediction[0][index] * 100)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"{emotion}  {confidence:.1f}%",
                (x, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        current_time = time.time()
        elapsed = current_time - self.last_time

        if elapsed > 0:
            self.fps = 1 / elapsed

        self.last_time = current_time

        self.emotion_label.configure(text=emotion)
        self.confidence_label.configure(
            text=f"{confidence:.1f}%"
        )
        self.fps_label.configure(
            text=f"{self.fps:.1f}"
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)

        width = self.camera_frame.winfo_width() - 20
        height = self.camera_frame.winfo_height() - 20

        image.thumbnail((width, height))

        photo = ImageTk.PhotoImage(image)

        self.video_label.configure(
            image=photo,
            text=""
        )

        self.video_label.image = photo

        self.root.after(10, self.update_frame)

    def close_application(self):
        self.running = False

        if self.webcam is not None:
            self.webcam.release()

        self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    app = EmotionApp(root)

    root.protocol(
        "WM_DELETE_WINDOW",
        app.close_application
    )

    root.mainloop()
