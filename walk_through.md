Here's what a new user needs to know to clone and run AgroAI locally:
# 🌱 AgroAI — Crop Disease Detection

This guide walks you through cloning, setting up, and running AgroAI locally.

---

## 1. Clone the Repository

Open your terminal and run:

```bash
git clone https://github.com/sushmas-wq/eage_AI_plant.git
cd eage_AI_plant

## 2. Set up a clean Python environment. Before installing anything, create a virtual environment so the dependencies don't conflict with other projects on your machine: python -m venv venv then source venv/bin/activate (Mac/Linux) or venv\Scripts\activate (Windows). You'll see (venv) appear in your terminal — that means it's active.
## 3. Install the dependencies. Run pip install -r requirements.txt. This pulls in FastAPI, PyTorch, OpenCV, and everything else the project needs. It may take a few minutes the first time.
## 4. Get the model checkpoints. The trained AI models are not included in the repo — contact the project author to get them. Once you have them, create a folder called checkptc in the project root and place all the .pth files inside it. Without this step the server will refuse to start.
## 5. Start the server. Run uvicorn app:app --host 0.0.0.0 --port 8000 --reload. Watch the terminal output. You'll see a line that says Crop model ready. — that's your signal that the AI is loaded and the server is accepting requests. If you see a FileNotFoundError instead, your checkptc/ folder is missing or the checkpoint filenames don't match what the code expects.
## 6. Confirm everything is working. Open your browser and go to http://localhost:8000/health. You should see something like this:

json   {"status": "ok", "device": "cpu", "crop_model_loaded": true}
If crop_model_loaded is false, the server is still initialising — wait a few seconds and refresh.

## 7. Find a leaf image to test with. Take a clear photo of a plant leaf, or find one online. The best results come from photos where the leaf fills most of the frame, taken in natural daylight with a plain background. Save it as a JPEG or PNG — other formats will be rejected.
## 8. Send your first request. You can do this two ways. The easiest is to go to http://localhost:8000/docs in your browser — this opens the interactive API docs. Click on POST /predict, then Try it out, upload your image file, and hit Execute. You'll see the full response right there without writing a single line of code.
Alternatively, call it from Python. Create a small script:

python   import requests

   with open("leaf.jpg", "rb") as f:
       response = requests.post(
           "http://localhost:8000/predict",
           files={"file": ("leaf.jpg", f, "image/jpeg")}
       )

   print(response.json())
Run it with python your_script.py.

## 9. Read the response. A successful result looks like this:

json    {
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
## 10. Here's how to interpret each field — `crop` tells you which plant the model identified. `label` is the clean disease name. `severity` is the percentage of the leaf area that appears diseased — a value under 10% is early stage, above 40% is severe. `disease_confidence` tells you how sure the model is on a scale of 0 to 1; anything above 0.7 is a reliable result. The `insight` block gives you the cause and what to do about it.
## 11 Understand what an uncertain result means. If the model isn't confident enough it returns this instead:
json    {
      "disease": "uncertain",
      "below_threshold": true,
      "insight": {
        "advice": "Retake in good natural lighting with the leaf filling the frame."
      }
    }
This is not an error — it's the model being honest. Retake the photo with better lighting, get closer to the leaf, and try again.
## 12. Get a visual result. If you want to see exactly which parts of the leaf the model flagged as diseased, send the same image to /visualize instead of /predict:
python   
    
    import requests, base64
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
The saved image will show the leaf with diseased pixels highlighted in red. This is useful for understanding how widespread the infection is and whether it matches what you can see with your eyes.
## 13. You're done. At this point you've cloned the project, got it running, sent a real leaf image, received a diagnosis, interpreted the confidence and severity scores, and seen the visual overlay. That's the full capability of the application — everything else in the codebase is the machinery that makes those seven response fields possible.
