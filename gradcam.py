import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---------------- PATHS ----------------
MODEL_PATH = "models/resnet50_melanoma_final.keras"
IMG_PATH = "test.jpg"
IMG_SIZE = 224

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- LOAD IMAGE ----------------
img = cv2.imread(IMG_PATH)
if img is None:
    raise ValueError("Image not found")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

input_img = preprocess_input(img.copy())
input_img = np.expand_dims(input_img,axis=0)

# ---------------- GRAD CAM ----------------
last_conv_layer = model.get_layer("conv4_block6_out")

grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[last_conv_layer.output, model.output]
)

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(input_img)

    print("Model Prediction:", predictions.numpy())

    loss = predictions[:,0]

grads = tape.gradient(loss, conv_outputs)

conv_outputs = conv_outputs[0]
grads = grads[0]

weights = tf.reduce_mean(grads, axis=(0,1))

cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

for i, w in enumerate(weights):
    cam += w * conv_outputs[:,:,i]

cam = np.maximum(cam,0)

if cam.max() != 0:
    cam = cam / cam.max()

cam = cv2.resize(cam,(IMG_SIZE,IMG_SIZE))

# ---------------- HEATMAP ----------------
heatmap = np.uint8(255*cam)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)

overlay = cv2.addWeighted(img,0.7,heatmap,0.5,0)

# ==================================================
# ABCDE MELANOMA ANALYSIS
# ==================================================

# create lesion mask
threshold = 0.3
mask = (cam > threshold).astype("uint8")

kernel = np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# -------- Asymmetry --------
h,w = mask.shape
left = mask[:,:w//2]
right = np.fliplr(mask[:,w//2:])

A_score = np.sum(np.abs(left-right[:,:left.shape[1]]))/(h*w)

if A_score > 0.10:
    A_text = "Asymmetry detected in lesion structure"
else:
    A_text = "Lesion appears relatively symmetric"

# -------- Border --------
contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    c = max(contours,key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c,True)
    circularity = (4*np.pi*area)/(perimeter**2 + 1e-6)
else:
    circularity = 1

if circularity < 0.6:
    B_text = "Irregular lesion borders detected"
else:
    B_text = "Border appears smooth"

# -------- Color --------
pixels = img[mask==1]

if len(pixels) > 0:
    color_std = np.std(pixels)
else:
    color_std = 0

if color_std > 40:
    C_text = "Multiple color variations present"
else:
    C_text = "Color distribution relatively uniform"

# -------- Diameter --------
area_ratio = np.sum(mask)/(IMG_SIZE*IMG_SIZE)

if area_ratio > 0.20:
    D_text = "Large lesion size relative to image"
else:
    D_text = "Lesion size relatively small"

# -------- Evolution --------
E_text = "Evolution cannot be determined from single image"

# ---------------- PRINT ABCDE ----------------
print("\n------ ABCDE CLINICAL INTERPRETATION ------")
print("A (Asymmetry):",A_text)
print("B (Border):",B_text)
print("C (Color):",C_text)
print("D (Diameter):",D_text)
print("E (Evolution):",E_text)
print("--------------------------------------------")

# ---------------- DISPLAY ----------------
plt.figure(figsize=(9,3))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM")
plt.imshow(cam,cmap="jet")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.savefig("gradcam_result.png",dpi=300)
plt.show()

print("\nGradCAM result saved as gradcam_result.png")