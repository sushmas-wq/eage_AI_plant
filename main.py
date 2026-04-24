import io
import logging
import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from models import load_crop_model, load_disease_model, DEVICE
from inference import run_full_pipeline, build_overlay
from utils import bgr_to_base64
from fastapi.middleware.cors import CORSMiddleware
# ============================


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App state (models loaded once at startup) ─────────────────────────────────
class AppState:
    crop_model   = None
    crop_classes = None
    # Disease models cached per crop so we load each once per process lifetime
    _disease_cache: dict = {}

    @classmethod
    def get_disease_model(cls, crop: str):
        if crop not in cls._disease_cache:
            logger.info("Loading disease model for crop '%s'", crop)
            model, classes = load_disease_model(crop)
            cls._disease_cache[crop] = (model, classes)
        return cls._disease_cache[crop]


# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — device: %s", DEVICE)
    AppState.crop_model, AppState.crop_classes = load_crop_model()
    logger.info("Crop model ready.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgroAI API",
    description="Crop disease detection — segmentation, classification, severity.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ── Helpers ───────────────────────────────────────────────────────────────────
async def read_image(file: UploadFile) -> Image.Image:
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))  # keep original object
        fmt = img.format  # store format BEFORE convert
        img = img.convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Upload a valid JPEG or PNG file."
        )

    if fmt not in ("JPEG", "PNG"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{fmt}'. Upload JPEG or PNG."
        )

    return img

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "crop_model_loaded": AppState.crop_model is not None,
    }


@app.post("/predict", tags=["inference"])
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Full inference pipeline.
    Returns crop, disease, severity, label and actionable insight.
    """
    img_pil = await read_image(file)

    if AppState.crop_model is None:
        raise HTTPException(status_code=503, detail="Crop model not loaded.")

    try:
        result = await asyncio.to_thread(
            run_full_pipeline,
            img_pil,
            AppState.crop_model,
            AppState.crop_classes,
            AppState.get_disease_model,     # callable(crop) -> (model, classes)
        )
    except ValueError as exc:
        logger.warning("Prediction error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal inference error.") from exc

    return JSONResponse(content=result)


@app.post("/visualize", tags=["inference"])
async def visualize_endpoint(file: UploadFile = File(...)):
    """
    Returns base64-encoded segmented image and disease overlay.
    """
    img_pil = await read_image(file)

    if AppState.crop_model is None:
        raise HTTPException(status_code=503, detail="Crop model not loaded.")

    try:
        segmented_bgr, overlay_bgr = await asyncio.to_thread(
            build_overlay,
            img_pil,
            AppState.crop_model,
            AppState.crop_classes,
            AppState.get_disease_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during visualization")
        raise HTTPException(status_code=500, detail="Internal visualization error.") from exc

    return JSONResponse(content={
        "segmented_image_b64": bgr_to_base64(segmented_bgr),
        "overlay_image_b64":   bgr_to_base64(overlay_bgr),
    })
