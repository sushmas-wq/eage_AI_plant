import cv2
import numpy as np
import re
from skimage.feature import local_binary_pattern
from torchvision import transforms
from PIL import Image

# ── Transform ────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ── Leaf segmentation ─────────────────────────────────────────────────────────
def segment_leaf(img_bgr, min_area=4000):
    max_size = 512
    h, w = img_bgr.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_green  = np.array([25, 20, 20]);   upper_green  = np.array([95, 255, 255])
    lower_yellow = np.array([15, 40, 40]);   upper_yellow = np.array([35, 255, 255])
    lower_brown  = np.array([5, 50, 20]);    upper_brown  = np.array([20, 255, 200])
    lower_red1   = np.array([0, 50, 50]);    upper_red1   = np.array([10, 255, 255])
    lower_red2   = np.array([170, 50, 50]);  upper_red2   = np.array([180, 255, 255])
    lower_dark   = np.array([0, 0, 0]);      upper_dark   = np.array([180, 255, 60])
    lower_white  = np.array([0, 0, 180]);    upper_white  = np.array([180, 40, 255])

    mask_green  = cv2.inRange(hsv, lower_green,  upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_brown  = cv2.inRange(hsv, lower_brown,  upper_brown)
    mask_red1   = cv2.inRange(hsv, lower_red1,   upper_red1)
    mask_red2   = cv2.inRange(hsv, lower_red2,   upper_red2)
    mask_dark   = cv2.inRange(hsv, lower_dark,   upper_dark)
    mask_white  = cv2.inRange(hsv, lower_white,  upper_white)

    seed = (mask_green | mask_yellow | mask_brown |
            mask_red1  | mask_red2   | mask_dark  | mask_white)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    seed   = cv2.morphologyEx(seed, cv2.MORPH_OPEN,  kernel)
    seed   = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel)

    h, w = seed.shape
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[seed > 0] = cv2.GC_PR_FGD
    gc_mask[h//4:3*h//4, w//4:3*w//4][seed[h//4:3*h//4, w//4:3*w//4] > 0] = cv2.GC_FGD
    gc_mask[:10, :]  = cv2.GC_BGD
    gc_mask[-10:, :] = cv2.GC_BGD
    gc_mask[:, :10]  = cv2.GC_BGD
    gc_mask[:, -10:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model,
                iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

    mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype("uint8")

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask  = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(final_mask, [c], -1, 255, -1)

    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=final_mask)
    return final_mask, segmented


# ── Disease masks ─────────────────────────────────────────────────────────────
def detect_white_fungus(segmented_bgr, leaf_mask):
    hsv = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    white_mask  = cv2.inRange(hsv, lower_white, upper_white)
    white_mask  = cv2.bitwise_and(white_mask, white_mask, mask=leaf_mask)
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_mask  = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN,  kernel)
    white_mask  = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    return (white_mask > 0).astype(np.uint8)


def disease_color_mask(segmented, leaf_mask):
    hsv          = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
    lower_brown  = np.array([8, 80, 50]);   upper_brown  = np.array([25, 255, 200])
    lower_yellow = np.array([18, 80, 80]);  upper_yellow = np.array([35, 255, 255])
    brown        = cv2.inRange(hsv, lower_brown,  upper_brown)
    yellow       = cv2.inRange(hsv, lower_yellow, upper_yellow)
    color_mask   = cv2.bitwise_or(brown, yellow)
    color_mask   = cv2.bitwise_and(color_mask, color_mask, mask=leaf_mask)
    return (color_mask > 0).astype(np.uint8)


def disease_texture_mask(segmented, leaf_mask, color_mask):
    gray     = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    lbp      = local_binary_pattern(gray, 16, 2, method="uniform")
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    _, texture = cv2.threshold(lbp_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    texture  = cv2.bitwise_and(texture, texture, mask=color_mask)
    texture  = cv2.bitwise_and(texture, texture, mask=leaf_mask)
    return (texture > 0).astype(np.uint8)


def final_disease_mask(segmented, leaf_mask):
    color        = disease_color_mask(segmented, leaf_mask)
    texture      = disease_texture_mask(segmented, leaf_mask, color)
    core_disease = np.logical_and(color, texture)
    white_fungus = detect_white_fungus(segmented, leaf_mask)
    disease      = np.logical_or(core_disease, white_fungus).astype(np.uint8)
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease      = cv2.morphologyEx(disease, cv2.MORPH_OPEN,  kernel)
    disease      = cv2.morphologyEx(disease, cv2.MORPH_CLOSE, kernel)
    return disease


# ── Severity ──────────────────────────────────────────────────────────────────
def compute_severity(disease_mask, leaf_mask):
    diseased_pixels = np.sum(disease_mask)
    leaf_pixels     = np.sum(leaf_mask > 0)
    return (diseased_pixels / (leaf_pixels + 1e-7)) * 100


# ── Label helpers ─────────────────────────────────────────────────────────────
def extract_label_parts(label: str) -> str:
    if "___" in label:
        disease = label.split("___", 1)[1]
    elif "__" in label:
        disease = label.split("__", 1)[1]
    elif "_" in label:
        disease = label.split("_", 1)[1]
    else:
        disease = label

    if re.search(r"healthy", disease, re.IGNORECASE):
        return "healthy"

    disease = disease.replace("_", " ").strip()
    disease = re.sub(r"\s+", " ", disease)
    return disease


# ── Image encoding ────────────────────────────────────────────────────────────
def bgr_to_base64(img_bgr: np.ndarray) -> str:
    import base64
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buf).decode("utf-8")


def pil_to_bgr(img_pil) -> np.ndarray:
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)