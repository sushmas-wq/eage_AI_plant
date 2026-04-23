import streamlit as st
import requests
from PIL import Image
import base64
import io
# ===========================
#  INFO
# ============================
infoa = {
    "Healthy": {
        "title": "Plant appears healthy 🌱",
        "cause": "No visible disease symptoms detected.",
        "advice": "Continue regular monitoring and proper irrigation."
    },
    "Apple Scab":{
        "title": "Apple Scab detected 🍏",
        "cause": "Fungal disease causing dark lesions on leaves and fruit.",
        "advice": "Remove infected material and apply fungicide."
    },
    "Black Rot":{
        "title": "Black Rot detected 🍎",
        "cause": "Fungal infection leading to black lesions on leaves and fruit.",
        "advice": "Prune affected areas and use appropriate fungicides."
    },
    "Cedar Apple Rust":{
        "title": "Cedar Apple Rust detected 🌿",
        "cause": "Fungal disease requiring both apple and cedar hosts.",
        "advice": "Remove nearby cedar trees and apply fungicide."
    },
    "Cercospora Leaf Spot":{
        "title": "Cercospora Leaf Spot detected 🌽",
        "cause": "Fungal infection causing circular spots on leaves.",
        "advice": "Improve air circulation and apply fungicide."
    },
    "Common Rust": {
        "title": "Common Rust detected 🌽",
        "cause": "Fungal disease producing rust-colored pustules on leaves.",
        "advice": "Use resistant varieties and apply fungicide if needed."
    },"Esca (Black Measles)": {
        "title": "Esca (Black Measles) detected 🍇",
        "cause": "Fungal disease affecting grapevines.",
        "advice": "Remove infected vines and apply appropriate fungicides."
    },"Haunglongbing (Citrus greening)": {
        "title": "Haunglongbing (Citrus greening) detected  🍊",
        "cause": "Bacterial disease spread by psyllid insects.",
        "advice": "Remove infected trees and control psyllid population."
    },"Tomato Yellow Leaf Curl Virus": {
        "title": "Tomato Yellow Leaf Curl Virus detected 🍅",
        "cause": "Viral disease transmitted by whiteflies.",
        "advice": "Control whitefly population and remove infected plants."
    },"Brown Rust": {
        "title": "Brown Rust detected 🌾",
        "cause": "Fungal disease producing brown pustules on wheat leaves.",
        "advice": "Use resistant varieties and apply fungicide if needed."
    },
    "Leaf Rust": {
        "title": "Leaf Rust detected 🍂",
        "cause": "Fungal disease spread by airborne spores in humid conditions.",
        "advice": "Remove infected leaves and apply fungicide if needed."
    },
    "Powdery mildew": {
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
    "Late Blight":{
          "title": "Late Blight detected 🌧️",
        "cause": "Fungal disease thriving in cool, wet conditions.",
        "advice": """
<ul>
  <li>Remove and destroy infected plant material immediately to reduce spread</li>
  <li>Apply a bio-enzyme / microbial formulation (such as enzyme-based or beneficial microbe solutions)</li>
  <li>Ensure good air circulation and avoid overhead irrigation</li>
  <li>Apply preventively spray during high-humidity periods for best results</li>
  <li>👉 2–3 ml  of bio-enzyme per liter of water</li>
  <li>Spray every 10–14 days</li>
</ul>
"""
    },
    "Early Blight":{
        "title": "Early Blight detected 🌞",
        "cause": "Fungal disease favored by warm, dry weather.",
        "advice": "Remove infected material and apply fungicide."
    },
    "Tungro":{
        "title": "Tungo detected 🌾",
        "cause": "Caused by Rice Tungro Virus transmitted by green leafhoppers.",
        "advice": "Remove infected plants, control leafhopper population, and use resistant rice varieties."
},
}   
# ------------------------
# CONFIG
# ------------------------
API_URL = "https://eage-ai-plant.onrender.com/predict"  # change after deployment

st.set_page_config(
    page_title="AgroAI 🌱",
    page_icon="🌿",
    layout="centered"
)

# ------------------------
# UI HEADER
# ------------------------
st.markdown("""
<h1 style='text-align: center; color: green;'>🌱 AgroAI</h1>
<p style='text-align: center;'>AI-powered crop & disease detection</p>
""", unsafe_allow_html=True)

# ------------------------
# FILE UPLOAD
# ------------------------
uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "png"])

# ------------------------
# MAIN LOGIC
# ------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Analysis"):
        with st.spinner("Analyzing via AI backend..."):

            try:
                # Send image to FastAPI
                file_bytes = uploaded_file.getvalue()

                files = {
                  "file": (
                     uploaded_file.name,
                      file_bytes,
                     "image/jpeg"  # force type (safe fix)
                        )
                        }

                response = requests.post(API_URL, files=files)

                if response.status_code != 200:
                    st.error("❌ API Error")
                    st.write(response.text)
                    st.stop()

                result = response.json()

                # ------------------------
                # DISPLAY RESULTS
                # ------------------------
                st.success(f"🌱 Crop: {result['crop']}")
                disease = result["disease"].replace("_", " ")[1:]
                st.error(f"🦠 Disease: {disease}")
                st.info(f"🩺 Severity: {result['severity']:.2f}%")
                st.markdown(f"### 🧾 Info: {infoa.get(result['disease'], {'title': 'No info available', 'cause': '', 'advice': ''})['title']}")

                # ------------------------
                # HEALTHY CHECK
                # ------------------------
                if "healthy" in result["disease"].lower():
                    st.success("✅ Healthy leaf")
                    st.stop()

                # ------------------------
                # OVERLAY IMAGE
                # ------------------------
                if "overlay" in result:
                    img_bytes = base64.b64decode(result["overlay"])
                    overlay_img = Image.open(io.BytesIO(img_bytes))
                    st.image(overlay_img, caption="Disease Highlight", use_container_width=True)


            except Exception as e:
                st.error("Something went wrong")
                st.write(e)
