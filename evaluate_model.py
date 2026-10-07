import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- PATHS ----------------
MODEL_PATH = "resnet50_melanoma_final.keras"
DATA_PATH = "final_dataset/test"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- LOAD TEST DATA ----------------
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_gen = datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# ---------------- PREDICTIONS ----------------
y_true = test_gen.classes
y_pred_prob = model.predict(test_gen)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# ---------------- METRICS ----------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n===== MODEL EVALUATION =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# Save metrics to file
with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
    f.write(f"Accuracy : {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall   : {recall}\n")
    f.write(f"F1 Score : {f1}\n")

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(y_true, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

print("\nResults saved inside 'results/' folder")