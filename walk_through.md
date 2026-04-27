# 🌱 AgroAI — Crop Disease Detection Walkthrough

This guide walks you through cloning, setting up, and using AgroAI locally from scratch.

---

## 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/sushmas-wq/eage_AI_plant.git
cd eage_AI_plant
```

---

## 2. Set Up a Clean Python Environment

Create a virtual environment so the dependencies don't conflict with other projects on your machine:

```bash
# Mac/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

You'll see `(venv)` appear in your terminal — that means it's active.

---

## 3. Install the Dependencies

```bash
pip install -r requirements.txt
```

This pulls in FastAPI, PyTorch, OpenCV, and everything else the project needs. It may take a few minutes the first time.

---

## 4. Get the Model Checkpoints

> ⚠️ The trained AI models are **not included in the repo** — contact the project author to get them.

Once you have them:

1. Create a folder called `checkptc` in the project root
2. Place all `.pth` files inside it

```
eage_AI_plant/
├── checkptc/
│   ├── best_crop_model.pth
│   ├── best_disease_Apple.pth
│   ├── best_disease_Tomato.pth
│   └── ...
├── app.py
├── models.py
└── ...
```

Without this step the server will refuse to start.

---

## 5. Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Watch the terminal output. You'll see:

```
Crop model ready.
```

That's your signal that the AI is loaded and the server is accepting requests.

> ❌ If you see a `FileNotFoundError` instead, your `checkptc/` folder is missing or the checkpoint filenames don't match what the code expects.

---

## 6. Confirm Everything is Working

Open your browser and go to:

```
http://localhost:8000/health
```

You should see:

```json
{"status": "ok", "device": "cpu", "crop_model_loaded": true}
```

> If `crop_model_loaded` is `false`, the server is still initialising — wait a few seconds and refresh.

---

## 7. Find a Leaf Image to Test With

Take a clear photo of a plant leaf, or find one online. For the best results:

- The leaf should fill most of the frame
- Use natural daylight
- Aim for a plain or simple background
- Save it as **JPEG or PNG** — other formats will be rejected

---

## 8. Send Your First Request

**Option A — No code needed (recommended for first-timers):**

Go to `http://localhost:8000/docs` in your browser. This opens the interactive API docs. Click on `POST /predict` → `Try it out` → upload your image → `Execute`. You'll see the full response right there.

**Option B — Python script:**

```python
import requests

with open("leaf.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": ("leaf.jpg", f, "image/jpeg")}
    )

print(response.json())
```

Run it with:

```bash
python your_script.py
```

---

## 9. Read the Response

A successful result looks like this:

```json
{
  "crop": "Tomato",
  "crop_confidence": 0.9714,
  "disease": "Tomato___Late_blight",
  "disease_confidence": 0.8832,
  "severity": 23.41,
  "label": "Late Blight",
  "insight": {
    "title": "Late Blight detected 🌧️",
    "cause": "Fungal disease thriving in cool, wet conditions.",
    "advice": "Remove and destroy infected plant material immediately..."
  }
}
```

Here's what each field means:

| Field | What it tells you |
|---|---|
| `crop` | Which plant the model identified |
| `label` | Clean, human-readable disease name |
| `severity` | % of the leaf area that appears diseased. Under 10% = early stage, above 40% = severe |
| `disease_confidence` | How sure the model is (0–1). Above 0.7 is a reliable result |
| `insight.cause` | What causes this disease |
| `insight.advice` | What to do about it |

---

## 10. Understand What an Uncertain Result Means

If the model isn't confident enough, it returns this instead:

```json
{
  "disease": "uncertain",
  "below_threshold": true,
  "insight": {
    "advice": "Retake in good natural lighting with the leaf filling the frame."
  }
}
```

This is **not an error** — it's the model being honest. Retake the photo with better lighting, get closer to the leaf, and try again.

---

## 11. Get a Visual Result

To see exactly which parts of the leaf the model flagged as diseased, send the same image to `/visualize`:

```python
import requests
import base64
from PIL import Image
from io import BytesIO

with open("leaf.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/visualize",
        files={"file": ("leaf.jpg", f, "image/jpeg")}
    )

data = response.json()
overlay = Image.open(BytesIO(base64.b64decode(data["overlay_image_b64"])))
overlay.save("overlay.jpg")
overlay.show()
```

The saved image will show the leaf with **diseased pixels highlighted in red**. This helps you understand how widespread the infection is and whether it matches what you can see with your eyes.

---

## 12. You're Done

At this point you have:

- ✅ Cloned the project and set up the environment
- ✅ Started the server and confirmed it's healthy
- ✅ Sent a real leaf image and received a diagnosis
- ✅ Interpreted the confidence and severity scores
- ✅ Seen the visual overlay of affected areas

That's the full capability of the application — everything else in the codebase is the machinery that makes those response fields possible.

---

## Quick Reference

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Check server status |
| `/predict` | POST | Get crop, disease, severity, and advice |
| `/visualize` | POST | Get segmented + disease overlay images |
| `/docs` | GET | Interactive API explorer |

**Live API:** `https://eage-ai-plant.onrender.com`  
> Note: the free-tier Render deployment may take 30–60 seconds to respond if it has been idle.
