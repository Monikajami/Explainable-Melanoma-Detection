import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------- SETTINGS ----------------
MODEL_PATH = "resnet50_melanoma_final.keras"
IMG_SIZE = 224
MELANOMA_THRESHOLD = 0.35   # 🔥 SAFER MEDICAL THRESHOLD

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

def predict_image(img_path):

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Image not found")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    prob = float(model.predict(img, verbose=0)[0][0])
    prob_percent = round(prob * 100, 2)

    print("\n" + "="*60)
    print("          MELANOMA DETECTION RESULT")
    print("="*60)
    print(f"Melanoma Probability: {prob_percent}% ({round(prob,4)})")
    print("="*60)

    # ================= PRIMARY DIAGNOSIS =================
    if prob >= MELANOMA_THRESHOLD:
        diagnosis = "⚠️ MELANOMA SUSPECTED"
        recommendation_main = "Dermatologist consultation strongly recommended."
    else:
        diagnosis = "✅ LIKELY BENIGN"
        recommendation_main = "Routine monitoring advised."

    print(f"\n🩺 Primary Diagnosis: {diagnosis}")
    print(f"Recommendation: {recommendation_main}")

    # ================= RISK STRATIFICATION =================
    if prob < 0.20:
        stage = 1
        risk = "Very Low Risk"
        visual = "🟢 GREEN"
    elif prob < 0.35:
        stage = 2
        risk = "Low Risk"
        visual = "🟡 YELLOW"
    elif prob < 0.60:
        stage = 3
        risk = "Moderate Risk"
        visual = "🟠 ORANGE"
    else:
        stage = 4
        risk = "High Risk"
        visual = "🔴 RED"

    print(f"\n📊 Risk Stage: {stage}/4")
    print(f"Risk Level: {risk}")
    print(f"Status Indicator: {visual}")
    print("="*60 + "\n")


# ---------------- RUN ----------------
predict_image("test.jpg")
