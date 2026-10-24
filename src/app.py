import io
import threading
import json
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string, jsonify

import mediapipe as mp

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the trained PyTorch model (should be a torchscript or state_dict file)
MODEL_PATH = "models/sign_language_classifier.pth"

# Expected input size for the model (height, width)
MODEL_INPUT_SIZE = (224, 224)

# Mapping from model output indices to human‑readable sign labels
# This should be consistent with the training script.
INDEX_TO_LABEL = {
    0: "Hello",
    1: "Thank you",
    2: "Yes",
    3: "No",
    4: "I love you",
    # Add further mappings as needed
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def preprocess_hand_image(img: np.ndarray) -> torch.Tensor:
    """
    Resize, normalize and convert a BGR image (as returned by OpenCV) to a
    torch tensor ready for inference.

    Args:
        img: Hand region image in BGR format.

    Returns:
        Tensor of shape (1, 3, H, W) with values in [0, 1] and normalized
        using ImageNet statistics.
    """
    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img_resized = cv2.resize(img_rgb, MODEL_INPUT_SIZE)

    # Convert to float32 and scale to [0, 1]
    img_float = img_resized.astype(np.float32) / 255.0

    # Normalise with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std

    # HWC → CHW and add batch dimension
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
    return tensor


def extract_hand_bbox(landmarks: mp.solutions.hands.HandLandmark,
                      image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Compute a tight axis‑aligned bounding box around a hand using its
    MediaPipe landmarks.

    Args:
        landmarks: Normalised hand landmarks.
        image_shape: (height, width) of the original image.

    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates.
    """
    h, w = image_shape
    xs = [landmark.x * w for landmark in landmarks]
    ys = [landmark.y * h for landmark in landmarks]
    x_min = max(int(min(xs)) - 10, 0)
    y_min = max(int(min(ys)) - 10, 0)
    x_max = min(int(max(xs)) + 10, w)
    y_max = min(int(max(ys)) + 10, h)
    return x_min, y_min, x_max, y_max


# ---------------------------------------------------------------------------
# Video camera class
# ---------------------------------------------------------------------------

class VideoCamera:
    """
    Captures frames from the webcam, runs hand detection with MediaPipe,
    performs sign language classification with a PyTorch model, and
    stores the most recent translation.
    """

    def __init__(self, model_path: str):
        # Initialise webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open webcam (device 0).")

        # Load the PyTorch model
        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()

        # MediaPipe hand detector
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Shared state
        self._lock = threading.Lock()
        self._latest_text = ""

    def __del__(self):
        # Release resources
        if self.cap.isOpened():
            self.cap.release()
        self.mp_hands.close()

    def get_latest_text(self) -> str:
        """Thread‑safe getter for the most recent translation."""
        with self._lock:
            return self._latest_text

    def _update_latest_text(self, text: str) -> None:
        """Thread‑safe setter for the most recent translation."""
        with self._lock:
            self._latest_text = text

    def _predict_sign(self, hand_img: np.ndarray) -> Optional[str]:
        """
        Run the classifier on a cropped hand image.

        Returns:
            The predicted label if inference succeeds, otherwise None.
        """
        try:
            tensor = preprocess_hand_image(hand_img)
            with torch.no_grad():
                logits = self.model(tensor)
                pred_idx = int(logits.squeeze().argmax().item())
            return INDEX_TO_LABEL.get(pred_idx, "Unknown")
        except Exception as e:
            # In production you would log this exception.
            return None

    def get_frame(self) -> Optional[bytes]:
        """
        Capture a single frame, annotate it, run inference and return the
        JPEG‑encoded image bytes.

        Returns:
            JPEG bytes or None if a frame could not be captured.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Flip horizontally for a mirror‑like view
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks on the original frame
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
            )

            # Extract bounding box and run classification
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = extract_hand_bbox(
                hand_landmarks.landmark, (h, w)
            )
            hand_crop = frame[y_min:y_max, x_min:x_max]
            if hand_crop.size != 0:
                label = self._predict_sign(hand_crop)
                if label:
                    self._update_latest_text(label)
                    # Overlay the label on the frame
                    cv2.putText(
                        frame,
                        label,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

        # Encode frame as JPEG
        success, jpeg = cv2.imencode(".jpg", frame)
        if not success:
            return None
        return jpeg.tobytes()


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

# Initialise a single shared camera instance
camera = VideoCamera(MODEL_PATH)


def generate_frames():
    """Yield JPEG frames in multipart format for streaming."""
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/")
def index():
    """
    Render a minimal HTML page that displays the video stream and the
    current translation. JavaScript polls the '/translation' endpoint
    every second to update the displayed text.
    """
    html = """
    <!doctype html>
    <html lang="en">
    <head>
        <title>Real‑Time Sign Language Translator</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 20px; }
            #video { border: 2px solid #444; }
            #translation { margin-top: 15px; font-size: 1.5em; color: #006400; }
        </style>
    </head>
    <body>
        <h1>Real‑Time Sign Language Translator</h1>
        <img id="video" src="{{ url_for('video_feed') }}" width="640" height="480">
        <div id="translation">Loading...</div>

        <script>
            async function fetchTranslation() {
                const resp = await fetch('{{ url_for('translation') }}');
                if (resp.ok) {
                    const data = await resp.json();
                    document.getElementById('translation').innerText = data.text;
                }
            }
            // Poll every second
            setInterval(fetchTranslation, 1000);
            // Initial fetch
            fetchTranslation();
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route("/video_feed")
def video_feed():
    """Route that streams the webcam feed."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/translation")
def translation():
    """Return the latest recognised sign as JSON."""
    text = camera.get_latest_text()
    return jsonify({"text": text})


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Enable threaded mode so that the video capture and Flask server run
    # concurrently without blocking each other.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)