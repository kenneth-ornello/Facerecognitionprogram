import os
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from deepface import DeepFace

# ── Constants ────────────────────────────────────────────────────────────────
DEEPFACE_MODEL       = 'ArcFace'
DISTANCE_METRIC      = 'cosine'
CONFIDENCE_THRESHOLD = 0.45   # slightly relaxed to improve dark-skin recall

# ── Image Preprocessing (skin-tone aware) ────────────────────────────────────

def preprocess_face(img_bgr: np.ndarray) -> np.ndarray:
    """
    Improve recognition accuracy for ALL skin tones (dark, medium, fair) by:
      1. CLAHE on the L-channel (luminance) in LAB colour space
         → lifts shadows without blowing out highlights
      2. Mild bilateral filter  → smooths noise while preserving edges / pores
      3. Slight sharpening      → recovers edge detail lost in bilateral pass
      4. Gamma correction       → nonlinear brightening for under-exposed faces

    The result is returned as BGR (same as input) so it can be fed directly
    to DeepFace / OpenCV without any further conversion.
    """
    if img_bgr is None:
        return img_bgr

    # ── 1. CLAHE in LAB colour space ─────────────────────────────────────────
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Detect if the face region is dark (mean L < 100 out of 255)
    mean_l = float(np.mean(l))
    clip   = 3.0 if mean_l < 100 else 2.0   # stronger CLAHE for dark images
    clahe  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq   = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    img_out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # ── 2. Bilateral filter (noise reduction, edge-preserving) ───────────────
    img_out = cv2.bilateralFilter(img_out, d=9, sigmaColor=75, sigmaSpace=75)

    # ── 3. Unsharp mask (gentle sharpening) ──────────────────────────────────
    blur      = cv2.GaussianBlur(img_out, (0, 0), sigmaX=3)
    img_out   = cv2.addWeighted(img_out, 1.5, blur, -0.5, 0)

    # ── 4. Gamma correction for very dark frames ─────────────────────────────
    if mean_l < 80:
        gamma  = 1.4
        lut    = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                           for i in range(256)], dtype=np.uint8)
        img_out = cv2.LUT(img_out, lut)

    return img_out


def multi_exposure(img_bgr: np.ndarray):
    """
    Yield 3 variants of the frame: original, brightened, darkened.
    DeepFace is tried on each; the first confident hit wins.
    This helps when the face is significantly over- or under-exposed.
    """
    yield img_bgr                                       # original
    bright = cv2.convertScaleAbs(img_bgr, alpha=1.3, beta=20)
    yield bright                                        # brighter
    dark   = cv2.convertScaleAbs(img_bgr, alpha=0.8, beta=-10)
    yield dark                                          # darker


# ── UI Helpers ───────────────────────────────────────────────────────────────

def get_button(window, text, color, command, fg='white'):
    return tk.Button(
        window, text=text,
        activebackground="black", activeforeground="white",
        fg=fg, bg=color, command=command,
        height=2, width=20,
        font=('Helvetica bold', 20)
    )

def get_img_label(window):
    return tk.Label(window)

def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label

def get_entry_text(window):
    return tk.Text(window, height=2, width=15, font=("Arial", 32))

def msg_box(title, description):
    messagebox.showinfo(title, description)


# ── Smart Recognition ─────────────────────────────────────────────────────────

def recognize(img: np.ndarray, db_path: str) -> str:
    """
    Multi-strategy, multi-exposure face recognition.

    Pipeline:
      For each exposure variant (normal / bright / dark):
        - Preprocess (CLAHE + bilateral + sharpen + gamma)
        - Try ArcFace + RetinaFace  (most accurate)
        - Try ArcFace + OpenCV      (faster fallback)
        - If distance ≤ threshold → return roll number stem

    Returns:
      - roll-number stem string on match
      - 'unknown_person'   if a face was found but not in the DB
      - 'no_persons_found' if no face could be detected at all
    """
    strategies = [
        {'detector_backend': 'retinaface'},
        {'detector_backend': 'opencv'},
    ]

    for variant in multi_exposure(img):
        processed = preprocess_face(variant)

        for strategy in strategies:
            try:
                results = DeepFace.find(
                    img_path=processed,
                    db_path=db_path,
                    model_name=DEEPFACE_MODEL,
                    distance_metric=DISTANCE_METRIC,
                    enforce_detection=False,
                    silent=True,
                    **strategy
                )

                if not results or results[0].empty:
                    continue

                best     = results[0].iloc[0]
                dist_col = [c for c in best.index if 'distance' in c.lower()]
                distance = float(best[dist_col[0]]) if dist_col else 1.0

                if distance <= CONFIDENCE_THRESHOLD:
                    name       = os.path.splitext(os.path.basename(best['identity']))[0]
                    confidence = round((1 - distance) * 100, 1)
                    print(f"[DeepFace] ✓ Matched '{name}' | "
                          f"dist={distance:.3f} | conf={confidence}% | "
                          f"strategy={strategy['detector_backend']}")
                    return name

            except Exception as e:
                print(f"[DeepFace] Strategy {strategy} failed: {e}")
                continue

    # ── No confident match – check if any face was visible ───────────────────
    try:
        faces = DeepFace.extract_faces(
            img_path=preprocess_face(img),
            enforce_detection=True,
            detector_backend='opencv'
        )
        if faces:
            return 'unknown_person'
    except Exception:
        pass

    return 'no_persons_found'