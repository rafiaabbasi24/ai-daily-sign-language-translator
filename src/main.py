import argparse
import logging
import os
import pathlib
import sys
from typing import List, Tuple

import cv2
import flask
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, request
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------
# Configuration & Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sign_language_translator")

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------


class SignLanguageDataset(Dataset):
    """
    PyTorch Dataset for sign language data.

    Expected folder structure:
        root/
            class_names.txt               # one class name per line
            train/
                *.npz                     # each file contains 'features' (np.ndarray) and 'label' (int)
            val/
                *.npz
            test/
                *.npz
    """

    def __init__(self, root_dir: pathlib.Path, split: str = "train"):
        """
        Args:
            root_dir: Path to the dataset root.
            split: One of 'train', 'val', 'test'.
        """
        self.root_dir = pathlib.Path(root_dir)
        self.split = split
        self.data_dir = self.root_dir / split
        self.samples = list(self.data_dir.glob("*.npz"))
        if not self.samples:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")

        class_file = self.root_dir / "class_names.txt"
        if not class_file.is_file():
            raise FileNotFoundError(f"Class names file not found: {class_file}")
        self.class_names = [line.strip() for line in class_file.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample_path = self.samples[idx]
        with np.load(sample_path) as data:
            features = data["features"]  # shape: (63,)
            label = int(data["label"])
        features_tensor = torch.from_numpy(features.astype(np.float32))
        return features_tensor, label

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------


class SignLanguageMLP(nn.Module):
    """
    Simple multi‑layer perceptron for classifying hand‑landmark vectors.
    Input size = 21 landmarks * 3 coordinates = 63.
    """

    def __init__(self, input_dim: int = 63, hidden_dims: List[int] = [128, 64], num_classes: int = 26):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------
# Training / Evaluation
# ----------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Run a single training epoch and return average loss."""
    model.train()
    running_loss = 0.0
    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            logger.debug(
                f"Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}"
            )
    avg_loss = running_loss / len(dataloader)
    logger.info(f"Training epoch completed – Avg loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation/test set. Returns (loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Evaluation – Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4%}")
    return avg_loss, accuracy


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, path: pathlib.Path) -> None:
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: pathlib.Path, device: torch.device) -> int:
    """Load model checkpoint. Returns the epoch to resume from."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    logger.info(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch


# ----------------------------------------------------------------------
# MediaPipe Hand Landmark Extraction
# ----------------------------------------------------------------------


class HandLandmarkExtractor:
    """
    Wrapper around MediaPipe Hands to extract normalized landmark vectors.
    Returns a (63,) numpy array (21 landmarks * 3 coordinates).
    """

    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 1, min_detection_confidence: float = 0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        )

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Process an RGB image and return a flattened landmark vector.
        If no hand is detected, returns a zero vector.
        """
        if image.shape[2] != 3:
            raise ValueError("Input image must have 3 channels (RGB).")
        results = self.hands.process(image)
        if not results.multi_hand_landmarks:
            return np.zeros(63, dtype=np.float32)

        # Use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        flat = []
        for lm in hand_landmarks.landmark:
            flat.extend([lm.x, lm.y, lm.z])
        return np.array(flat, dtype=np.float32)


# ----------------------------------------------------------------------
# Flask API for Real‑Time Inference
# ----------------------------------------------------------------------


def create_flask_app(model: nn.Module, class_names: List[str], device: torch.device) -> Flask:
    """
    Build a Flask application exposing a /predict endpoint.
    The endpoint expects a multipart/form-data with a file field named "image".
    """
    app = Flask(__name__)
    extractor = HandLandmarkExtractor(static_image_mode=False)

    @app.route("/predict", methods=["POST"])
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file_storage = request.files["image"]
        file_bytes = np.frombuffer(file_storage.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = extractor.extract(rgb_img)  # (63,)
        tensor = torch.from_numpy(landmarks).unsqueeze(0).to(device)  # (1,63)

        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(prob))
            pred_label = class_names[pred_idx]

        response = {
            "prediction": pred_label,
            "confidence": float(prob[pred_idx]),
            "probabilities": {class_names[i]: float(p) for i, p in enumerate(prob)},
        }
        return jsonify(response)

    return app


# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real‑Time Sign Language Translator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data-root", type=pathlib.Path, required=True, help="Root directory of the dataset")
    train_parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--checkpoint-dir", type=pathlib.Path, default=pathlib.Path("./checkpoints"), help="Directory to save checkpoints")
    train_parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--data-root", type=pathlib.Path, required=True, help="Root directory of the dataset")
    eval_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    eval_parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to model checkpoint")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start Flask inference server")
    serve_parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="Path to model checkpoint")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for Flask server")
    serve_parser.add_argument("--port", type=int, default=5000, help="Port for Flask server")

    return parser.parse_args()


def main() -> None:
    set_seed()
    args = parse_arguments()
    device = get_device()
    logger.info(f"Using device: {device}")

    if args.command == "train":
        # Load dataset
        train_dataset = SignLanguageDataset(args.data_root, split="train")
        val_dataset = SignLanguageDataset(args.data_root, split="val")
        num_classes = train_dataset.num_classes
        logger.info(f"Number of classes: {num_classes}")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Model, loss, optimizer
        model = SignLanguageMLP(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        start_epoch = 0
        if args.resume:
            latest_ckpt = max(args.checkpoint_dir.glob("epoch_*.pt"), default=None)
            if latest_ckpt:
                start_epoch = load_checkpoint(model, optimizer, latest_ckpt, device)
                logger.info(f"Resuming from epoch {start_epoch}")

        for epoch in range(start_epoch, args.epochs):
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")
            train_one_epoch(model, train_loader, criterion, optimizer, device)
            evaluate(model, val_loader, device)

            ckpt_path = args.checkpoint_dir / f"epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, epoch + 1, ckpt_path)

        logger.info("Training completed.")

    elif args.command == "eval":
        test_dataset = SignLanguageDataset(args.data_root, split="test")
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = SignLanguageMLP(num_classes=test_dataset.num_classes).to(device)
        optimizer = optim.AdamW(model.parameters())  # dummy optimizer for loading checkpoint
        load_checkpoint(model, optimizer, args.checkpoint, device)

        evaluate(model, test_loader, device)

    elif args.command == "serve":
        # Load model and class names
        dummy_dataset = SignLanguageDataset(pathlib.Path("./dummy"), split="train")  # only to get class names
        # The dummy path will never be accessed because we only need class_names; we can safely load from checkpoint directory instead
        # Load class names from the checkpoint's sibling file if exists
        class_names_path = args.checkpoint.parent / "class_names.txt"
        if not class_names_path.is_file():
            logger.error(f"Class names file not found at {class_names_path}")
            sys.exit(1)
        class_names = [line.strip() for line in class_names_path.read_text().splitlines() if line.strip()]

        model = SignLanguageMLP(num_classes=len(class_names)).to(device)
        optimizer = optim.AdamW(model.parameters())
        load_checkpoint(model, optimizer, args.checkpoint, device)

        app = create_flask_app(model, class_names, device)
        logger.info(f"Starting Flask server at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)

    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
