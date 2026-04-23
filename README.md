CODE REVIEW REPORT
AgroAI Plant Disease Detection

Repository: github.com/sushmas-wq/eage_AI_plant
Review date: April 2026  |  Reviewer: Claude (Anthropic)

Executive Summary
AgroAI is a well-structured FastAPI backend for crop disease detection, built on PyTorch, OpenCV GrabCut segmentation, and MobileNetV3 classifiers. The architecture is clean and the separation of concerns across main.py / inference.py / utils.py / models.py is sound. The primary issues are a dead code block in main.py that silently swallows format validation, an import loop risk between inference.py and utils.py, missing confidence-gate enforcement in inference, CORS configured for unrestricted access, and two duplicate files (utils.py and utils.py.py) that indicate a workflow problem. None of these are blocking for a prototype, but three of them are blocking for production.

Area	Score	Summary
Architecture & structure	8 / 10	Clean module split, good lifespan pattern
Code correctness	6 / 10	Dead code block masks a real validation bug
Error handling	7 / 10	Good HTTPException use, but conf gate is passive
Security	4 / 10	CORS wildcard, no rate limiting, no auth
Performance	7 / 10	asyncio.to_thread correct; no request timeout
Dependency management	6 / 10	loguru listed but not used; versions good
Repository hygiene	5 / 10	Duplicate file, no README, no .env.example
Overall	6 / 10	Solid prototype, three fixes needed for prod
 
1. Security issues
1.1  CORS allows all origins (main.py)
allow_origins=["*"] means any website in the world can send requests to your API and read the response. This is acceptable during local development but must be tightened before any public deployment.

Replace wildcard with explicit origin list
import os

ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8501').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # set via env var in production
    allow_methods=['POST', 'GET'],
    allow_headers=['Content-Type'],
)

High	main.py	CORS wildcard permits cross-origin access from any domain


1.2  No request size limit
FastAPI has no default upload size cap. A client can send a multi-gigabyte file and exhaust server memory. Add a size check in read_image before calling Image.open.

Add at the start of read_image
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB

async def read_image(file: UploadFile) -> Image.Image:
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
            detail='File too large. Maximum size is 10 MB.')

High	main.py	No upload size limit — server can be OOM-killed by a large file


1.3  No rate limiting
The /predict endpoint runs CPU-heavy GrabCut segmentation and two neural network forward passes per request. Without rate limiting, a single client can saturate the server. Add a simple per-IP limiter via slowapi.

Minimal rate limiting with slowapi
pip install slowapi

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post('/predict')
@limiter.limit('10/minute')   # adjust to your expected load
async def predict_endpoint(request: Request, file: UploadFile = File(...)):
    ...

Medium	main.py	No rate limiting — inference endpoint can be easily DoS'd


 
2. Code quality

2.1  lru_cache imported but never used (main.py)
from functools import lru_cache appears at the top of main.py but is never called anywhere. The disease model caching is done correctly via AppState._disease_cache. Remove the unused import.

Low	main.py	Unused import: from functools import lru_cache


2.2  loguru listed in requirements but not used
requirements.txt includes loguru==0.7.3 but the codebase uses the standard library logging module throughout. Either use loguru (it is a good choice) or remove it from requirements to avoid a 10MB unnecessary dependency on deployed containers.

Low	requirements.txt	loguru listed as dependency but never imported


2.4  backend.py not reviewed
The repository contains a backend.py file that was not accessible for review. If it is a leftover from the original Streamlit monolith, it should be deleted. If it contains active logic, it should be consolidated into the appropriate module (main.py, inference.py, or utils.py) and not exist as a parallel entry point.

Info	backend.py	Purpose unclear — verify whether this file is active or a leftover


2.5  app.py not reviewed
Similarly, app.py exists alongside main.py. FastAPI projects should have a single entry point. If app.py is the Streamlit frontend, rename it to something explicit like frontend.py or streamlit_app.py to avoid confusion with FastAPI conventions where app.py is often the application factory.

Info	app.py	Ambiguous name alongside main.py — rename to frontend.py for clarity


 
3. Performance
3.1  No inference timeout
GrabCut with 5 iterations on a 512px image can take 2–8 seconds on CPU depending on image content. There is no timeout on the asyncio.to_thread call, so a pathological image could hold a thread indefinitely. Wrap the call with asyncio.wait_for.

Add a timeout to both endpoints
try:
    result = await asyncio.wait_for(
        asyncio.to_thread(run_full_pipeline, img_pil, ...),
        timeout=30.0   # seconds
    )
except asyncio.TimeoutError:
    raise HTTPException(status_code=504, detail='Inference timed out.')

Medium	main.py	No timeout on asyncio.to_thread — slow images can block threads indefinitely


3.2  GrabCut runs twice for /predict + /visualize
If a frontend calls /predict and then /visualize on the same image, segmentation and disease masking run twice. For the WhatsApp delivery path this is fine (one call per image), but for a web UI showing both results, consider returning the overlay from /predict directly or caching the segmentation result keyed by image hash.

Low	inference.py	Segmentation pipeline duplicated across /predict and /visualize calls


3.3  No model warmup
The first inference request after startup will be slow because PyTorch allocates CUDA memory and JIT-compiles kernels on the first forward pass. Add a warmup call in the lifespan function with a dummy tensor to absorb this cost at startup rather than on the first real user request.

Add to lifespan after model load
@asynccontextmanager
async def lifespan(app: FastAPI):
    AppState.crop_model, AppState.crop_classes = load_crop_model()

    # Warmup — absorb JIT cost at startup, not on first user request
    import torch
    dummy = torch.zeros(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        AppState.crop_model(dummy)
    logger.info('Warmup complete.')
    yield

Low	main.py	No warmup — first request takes 2–5x longer than subsequent requests


 
4. What is working well
These are genuine strengths that should be preserved as the codebase grows.

Lifespan pattern (main.py)
Using @asynccontextmanager for startup and shutdown correctly replaces the deprecated on_event decorator. Model loading happens once at startup with clear logging. AppState._disease_cache lazily loads per-crop disease models on first use — the right tradeoff given that not all crops will appear in every session.

asyncio.to_thread (main.py)
Wrapping the synchronous CPU-heavy pipeline in asyncio.to_thread is the correct pattern. It moves blocking work to a thread pool without blocking the event loop, allowing FastAPI to accept new connections while inference is running. This is non-trivial to get right and it is done correctly here.

Module separation
The split across main.py / models.py / inference.py / utils.py is well-designed. Each file has a clear single responsibility. The inference pipeline (run_full_pipeline, build_overlay) is cleanly separated from HTTP concerns (read_image, endpoint handlers). This makes the code testable and the ML logic portable.

GrabCut segmentation pipeline (utils.py)
The multi-mask seed approach (combining green, yellow, brown, red, dark, and white HSV ranges before initialising GrabCut) is robust for diverse leaf conditions. Forcing definite foreground in the centre quarter and definite background at image edges is the right prior for leaf-centred captures. The LBP texture mask gated by colour is a sensible combination that reduces false positives.

Defensive image reading
Opening the image from bytes (Image.open(io.BytesIO(contents))) rather than trusting the Content-Type header is the right approach. The Content-Type bug that caused the original 'Unsupported image type None' error in Streamlit is correctly handled on the backend side once the dead code block is fixed.

requirements.txt is clean
Package versions are pinned, Python 3.11 is specified in runtime.txt, and the file uses comments to group dependencies. The only issue is the unused loguru entry.
 
5. Prioritised fix list
Address in this order. Items 1–4 are required before any public deployment.

#	Severity	File	Fix
			
2	High	main.py	Add upload size limit (10 MB) before Image.open
3	High	main.py	Restrict CORS origins via environment variable
4	High	inference.py	Enforce CONF_THRESHOLD inside run_full_pipeline
5	Medium	main.py	Add asyncio.wait_for timeout (30s) on both endpoints
6	Medium	main.py	Add rate limiting via slowapi (10 req/min per IP)
7	Medium	inference.py	Move local cv2 / utils imports to module level
8	Medium	utils.py.py	Delete duplicate file
9	Low	main.py	Remove unused lru_cache import
10	Low	requirements.txt	Remove loguru or adopt it consistently
11	Low	main.py	Add model warmup call in lifespan
12	Info	backend.py	Confirm purpose — delete or consolidate
13	Info	app.py	Rename to frontend.py for clarity
14	Info	repo	Add README.md with setup and API docs


Bottom line
The architecture is clean and the ML pipeline is well-implemented. Fix the four high/critical issues (dead code, upload size, CORS, confidence gate) and this is production-ready for a pilot deployment. The remaining items are quality-of-life improvements that become important at scale.

