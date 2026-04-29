import logging
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image                  
from utils import transform, extract_label_parts

logger = logging.getLogger(__name__)

CONF_THRESHOLD = float(0.15)

DISEASE_INFO = {
    "Healthy": {
        "title": "Plant appears healthy 🌱",
        "cause": "No visible disease symptoms detected.",
        "advice": "Continue regular monitoring and proper irrigation."
    },
    "Apple Scab": {
        "title": "Apple Scab detected 🍏",
        "cause": "Fungal disease causing dark lesions on leaves and fruit.",
        "advice": "Remove infected material and apply fungicide."
    },
    "Black Rot": {
        "title": "Black Rot detected 🍎",
        "cause": "Fungal infection leading to black lesions on leaves and fruit.",
        "advice": "Prune affected areas and use appropriate fungicides."
    },
    "Cedar Apple Rust": {
        "title": "Cedar Apple Rust detected 🌿",
        "cause": "Fungal disease requiring both apple and cedar hosts.",
        "advice": "Remove nearby cedar trees and apply fungicide."
    },
    "Cercospora Leaf Spot": {
        "title": "Cercospora Leaf Spot detected 🌽",
        "cause": "Fungal infection causing circular spots on leaves.",
        "advice": "Improve air circulation and apply fungicide."
    },
    "Common Rust": {
        "title": "Common Rust detected 🌽",
        "cause": "Fungal disease producing rust-colored pustules on leaves.",
        "advice": "Use resistant varieties and apply fungicide if needed."
    },
    "Esca (Black Measles)": {
        "title": "Esca (Black Measles) detected 🍇",
        "cause": "Fungal disease affecting grapevines.",
        "advice": "Remove infected vines and apply appropriate fungicides."
    },
    "Haunglongbing (Citrus greening)": {
        "title": "Haunglongbing (Citrus greening) detected 🍊",
        "cause": "Bacterial disease spread by psyllid insects.",
        "advice": "Remove infected trees and control psyllid population."
    },
    "Tomato Yellow Leaf Curl Virus": {
        "title": "Tomato Yellow Leaf Curl Virus detected 🍅",
        "cause": "Viral disease transmitted by whiteflies.",
        "advice": "Control whitefly population and remove infected plants."
    },
    "Brown Rust": {
        "title": "Brown Rust detected 🌾",
        "cause": "Fungal disease producing brown pustules on wheat leaves.",
        "advice": "Use resistant varieties and apply fungicide if needed."
    },
    "Leaf Rust": {
        "title": "Leaf Rust detected 🍂",
        "cause": "Fungal disease spread by airborne spores in humid conditions.",
        "advice": "Remove infected leaves and apply fungicide if needed."
    },
    "Powdery Mildew": {
        "title": "Powdery Mildew detected 🌫️",
        "cause": "Fungus growing on leaf surfaces due to poor air circulation.",
        "advice": "Improve airflow and apply sulfur-based treatment."
    },
    "Bacterial Blight": {
        "title": "Bacterial Blight detected 🦠",
        "cause": "Bacterial infection spread via rain splash and tools.",
        "advice": "Use clean tools and disease-free seeds."
    },
    "Leaf Spot": {
        "title": "Leaf Spot detected 🔴",
        "cause": "Fungal or bacterial pathogens favored by wet leaves.",
        "advice": "Avoid overhead watering and improve drainage."
    },
    "Late Blight": {
        "title": "Late Blight detected 🌧️",
        "cause": "Fungal disease thriving in cool, wet conditions.",
        "advice": (
            "Remove and destroy infected plant material immediately. "
            "Apply a bio-enzyme / microbial formulation. "
            "Ensure good air circulation and avoid overhead irrigation. "
            "Spray 2–3 ml of bio-enzyme per litre of water every 10–14 days."
        )
    },
    "Early Blight": {
        "title": "Early Blight detected 🌞",
        "cause": "Fungal disease favored by warm, dry weather.",
        "advice": "Remove infected material and apply fungicide."
    },
    "Tungro": {
        "title": "Tungro detected 🌾",
        "cause": "Caused by Rice Tungro Virus transmitted by green leafhoppers.",
        "advice": "Remove infected plants, control leafhopper population, and use resistant varieties."
    },
}

FALLBACK_INFO = {
    "title": "Unknown condition",
    "cause": "Pattern does not match known diseases.",
    "advice": "Consult an agriculture expert."
}


def predict(img_pil, model, classes):
    """Unchanged from original — runs classification on a PIL image."""
    if isinstance(img_pil, np.ndarray):
        img_pil = Image.fromarray(img_pil)
    x = transform(img_pil).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
    idx = probs.argmax(1).item()
    return classes[idx], probs.max(1)[0]


def get_disease_info(label_key: str) -> dict:
    return DISEASE_INFO.get(label_key, FALLBACK_INFO)


def run_full_pipeline(img_pil, crop_model, crop_classes,
                      disease_model_loader) -> dict:
    """
    Orchestrates the full prediction pipeline.
    disease_model_loader: callable(crop) -> (model, classes)
    Returns the /predict response dict.
    """
    import cv2
    from utils import pil_to_bgr, segment_leaf, final_disease_mask, compute_severity

    img_bgr = pil_to_bgr(img_pil)

    # Segmentation
    leaf_mask, segmented = segment_leaf(img_bgr)

    # Disease pixel mask + severity
    d_mask   = final_disease_mask(segmented, leaf_mask)
    severity = compute_severity(d_mask, leaf_mask)

    # Crop classification
    crop, crop_conf = predict(segmented, crop_model, crop_classes)
    crop_conf_val   = float(crop_conf.item())

    # Disease classification
    disease_model, disease_classes = disease_model_loader(crop)
    if disease_model is None:
        raise ValueError(f"No disease model available for crop '{crop}'")

    disease, disease_conf = predict(img_pil, disease_model, disease_classes)
    disease_conf_val      = float(disease_conf.item())

    # Label + insight
    label     = extract_label_parts(disease)
    label_key = label.title()
    insight   = get_disease_info(label_key)
    disease_conf_val = float(disease_conf.item())

    # Enforce confidence gate
    if disease_conf_val < CONF_THRESHOLD:
      return {
        'crop': crop,
        'crop_confidence': round(crop_conf_val, 4),
        'disease': 'uncertain',
        'disease_confidence': round(disease_conf_val, 4),
        'severity': 0.0,
        'label': 'Uncertain',
        'insight': {
            'title': 'Low confidence result',
            'cause': 'The model is not confident about this image.',
            'advice': 'Retake in good natural lighting with the leaf filling the frame.'
        },
        'below_threshold': True
    }



    return {
        "crop":               crop,
        "crop_confidence":    round(crop_conf_val, 4),
        "disease":            disease,
        "disease_confidence": round(disease_conf_val, 4),
        "severity":           round(severity, 2),
        "label":              label_key,
        "insight":            insight,
    }


def build_overlay(img_pil, crop_model, crop_classes,
                  disease_model_loader) -> tuple:
    """
    Returns (segmented_bgr, overlay_bgr) for /visualize.
    """
    import cv2
    from utils import pil_to_bgr, segment_leaf, final_disease_mask

    img_bgr = pil_to_bgr(img_pil)
    leaf_mask, segmented = segment_leaf(img_bgr)
    d_mask               = final_disease_mask(segmented, leaf_mask)

    overlay = segmented.copy()
    overlay[d_mask == 1] = [0, 0, 255]   # red on BGR

    return segmented, overlay
