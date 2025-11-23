import os
from io import BytesIO
from typing import Dict, Any, List
from dotenv import load_dotenv

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Load environment
load_dotenv()

CV_MODEL_PATH = os.getenv("CV_MODEL_PATH", "models/disease_model/best_model.pth")
CV_CLASSES_PATH = os.getenv("CV_CLASSES_PATH", "models/disease_model/classes.txt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Preprocessing
INFER_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def load_class_names() -> List[str]:
    if not os.path.exists(CV_CLASSES_PATH):
        raise FileNotFoundError(f"Missing {CV_CLASSES_PATH}")
    with open(CV_CLASSES_PATH, "r") as f:
        return [x.strip() for x in f.readlines()]

def build_model(num_classes: int):
    # Load torchvision EfficientNet-B0
    model = efficientnet_b0(weights=None)  # do not load pretrained
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Load trained weights
    state_dict = torch.load(CV_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

_CLASS_NAMES = None
_MODEL = None

def _ensure_loaded():
    global _CLASS_NAMES, _MODEL
    if _CLASS_NAMES is None:
        _CLASS_NAMES = load_class_names()
    if _MODEL is None:
        _MODEL = build_model(len(_CLASS_NAMES))

def predict_from_pil(img: Image.Image, top_k: int = 3) -> Dict[str, Any]:
    _ensure_loaded()

    img = img.convert("RGB")
    x = INFER_TRANSFORMS(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _MODEL(x)
        probs = torch.softmax(logits, dim=1)[0]

    values, indices = torch.topk(probs, top_k)

    return {
        "label": _CLASS_NAMES[int(indices[0])],
        "confidence": float(values[0]),
        "topk": [( _CLASS_NAMES[int(idx)], float(val) ) for idx, val in zip(indices, values) ],
    }

def predict(image_file, top_k: int = 3):
    if hasattr(image_file, "read"):
        img = Image.open(image_file)
    elif isinstance(image_file, str):
        img = Image.open(image_file)
    else:
        img = Image.open(BytesIO(image_file))
    return predict_from_pil(img, top_k)
