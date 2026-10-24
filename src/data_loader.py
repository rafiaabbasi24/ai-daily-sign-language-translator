import cv2
import numpy as np
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

# MediaPipe is a heavy optional dependency; import lazily to provide a clear error message.
try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError(
        "mediapipe is required for hand landmark extraction. Install it with "
        "`pip install mediapipe`."
    ) from e


__all__ = [
    "HandDataLoader",
    "extract_hand_landmarks",
]


def extract_hand_landmarks(
    image: np.ndarray,
    *,
    static_image_mode: bool = False,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Extract normalized hand landmarks from an RGB image using MediaPipe.

    Parameters
    ----------
    image : np.ndarray
        The input image in BGR format (as read by OpenCV). It will be converted
        to RGB internally.
    static_image_mode : bool, optional
        If True, treats the input as a static image. Default is False (video mode).
    max_num_hands : int, optional
        Maximum number of hands to detect. Default is 2.
    min_detection_confidence : float, optional
        Minimum confidence value ([0.0, 1.0]) for hand detection to be considered
        successful. Default is 0.5.
    min_tracking_confidence : float, optional
        Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be tracked
        across frames. Default is 0.5.

    Returns
    -------
    Optional[np.ndarray]
        A ``(N, 63)`` array where ``N`` is the number of detected hands (0, 1 or 2)
        and each hand is represented by 21 landmarks * 3 coordinates (x, y, z)
        normalized to the image dimensions. Returns ``None`` if no hand is found.
    """
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as hands:
        # MediaPipe expects RGB images.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks: List[np.ndarray] = []
        for hand_landmark in results.multi_hand_landmarks:
            # Flatten (x, y, z) for each of the 21 landmarks.
            flattened = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmark.landmark],
                dtype=np.float32,
            ).flatten()
            hand_landmarks.append(flattened)

        return np.stack(hand_landmarks, axis=0)  # shape (N, 63)


class HandDataLoader:
    """
    Captures webcam frames, extracts hand landmarks using MediaPipe, and provides
    a rolling buffer of fixed‑length sequences suitable for real‑time inference.

    Example
    -------
    >>> loader = HandDataLoader(seq_len=30)
    >>> loader.start()
    >>> time.sleep(2)  # collect some frames
    >>> seq = loader.get_sequence()
    >>> if seq is not None:
    ...     # ``seq`` has shape (30, 63) – ready for a model
    ...     pass
    >>> loader.stop()
    """

    def __init__(
        self,
        seq_len: int = 30,
        device: int = 0,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        max_num_hands: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        seq_len : int, optional
            Length of the sliding window (number of frames). Default is 30.
        device : int, optional
            OpenCV video capture device index. Default is 0 (default webcam).
        detection_confidence : float, optional
            Minimum detection confidence for MediaPipe Hands. Default is 0.5.
        tracking_confidence : float, optional
            Minimum tracking confidence for MediaPipe Hands. Default is 0.5.
        max_num_hands : int, optional
            Maximum number of hands to detect per frame. Default is 1.
        """
        if seq_len <= 0:
            raise ValueError("seq_len must be a positive integer")
        if not (0.0 <= detection_confidence <= 1.0):
            raise ValueError("detection_confidence must be between 0 and 1")
        if not (0.0 <= tracking_confidence <= 1.0):
            raise ValueError("tracking_confidence must be between 0 and 1")
        if max_num_hands not in (1, 2):
            raise ValueError("max_num_hands must be 1 or 2")

        self.seq_len: int = seq_len
        self.device: int = device
        self.detection_confidence: float = detection_confidence
        self.tracking_confidence: float = tracking_confidence
        self.max_num_hands: int = max_num_hands

        self._capture: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Buffer stores ``seq_len`` frames of shape (63,) per hand.
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)

        # MediaPipe Hands object is created once per thread to avoid re‑initialisation.
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

    def start(self) -> None:
        """
        Starts the background thread that reads frames from the webcam,
        extracts landmarks, and fills the internal buffer.
        """
        if self._thread and self._thread.is_alive():
            raise RuntimeError("HandDataLoader is already running")

        self._capture = cv2.VideoCapture(self.device)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video device {self.device}")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Signals the background thread to stop, waits for it to finish,
        and releases the webcam resource.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if self._capture:
            self._capture.release()
        self._capture = None
        self._thread = None
        self._buffer.clear()

    def _run(self) -> None:
        """
        Internal method executed in a separate thread. Continuously reads frames,
        extracts hand landmarks, and appends them to the buffer.
        """
        assert self._capture is not None, "VideoCapture must be initialized before running"
        while not self._stop_event.is_set():
            ret, frame = self._capture.read()
            if not ret:
                # If a frame cannot be read, wait a short period before retrying.
                time.sleep(0.01)
                continue

            landmarks = self._process_frame(frame)
            if landmarks is not None:
                # If multiple hands are detected, we keep only the first one for simplicity.
                self._buffer.append(landmarks[0])
            else:
                # Append a zero‑vector to keep temporal alignment when no hand is detected.
                self._buffer.append(np.zeros(63, dtype=np.float32))

            # Small sleep to avoid hogging the CPU; the exact value can be tuned.
            time.sleep(0.001)

    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts hand landmarks from a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image captured from OpenCV.

        Returns
        -------
        Optional[np.ndarray]
            ``(N, 63)`` array where ``N`` is the number of detected hands.
            Returns ``None`` when no hand is detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = []
        for hand_landmark in results.multi_hand_landmarks:
            flattened = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmark.landmark],
                dtype=np.float32,
            ).flatten()
            hand_landmarks.append(flattened)

        return np.stack(hand_landmarks, axis=0)  # shape (N, 63)

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Retrieves the current sequence buffer.

        Returns
        -------
        Optional[np.ndarray]
            An array of shape ``(seq_len, 63)`` if the buffer is full,
            otherwise ``None``. The returned array is a copy and safe to modify.
        """
        if len(self._buffer) < self.seq_len:
            return None
        # Convert deque to a contiguous NumPy array.
        return np.stack(list(self._buffer), axis=0).astype(np.float32)

    def is_running(self) -> bool:
        """
        Returns ``True`` if the capture thread is active.

        Returns
        -------
        bool
            Running state of the loader.
        """
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()