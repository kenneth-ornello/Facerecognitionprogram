import os
import datetime
import csv
import io
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from deepface import DeepFace

# Suppress TensorFlow/oneDNN noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

# ── Config (all paths are local – no cloud storage) ──────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DB_DIR        = os.path.join(BASE_DIR, 'db')
LOG_PATH      = os.path.join(BASE_DIR, 'log.csv')
STUDENTS_JSON = os.path.join(BASE_DIR, 'students.json')

DEEPFACE_MODEL       = 'ArcFace'
DEEPFACE_BACKEND     = 'retinaface'
DISTANCE_METRIC      = 'cosine'
CONFIDENCE_THRESHOLD = 0.45   # slightly relaxed for better dark/fair recall
CSV_HEADER = ['Roll No', 'Name', 'Division', 'Department', 'Timestamp', 'Status']

# ── Boot-time setup ──────────────────────────────────────────────────────────
os.makedirs(DB_DIR, exist_ok=True)

if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(CSV_HEADER)

if not os.path.exists(STUDENTS_JSON):
    with open(STUDENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump({}, f)


# ── Face preprocessing (skin-tone aware) ─────────────────────────────────────

def preprocess_face(img_bgr: np.ndarray) -> np.ndarray:
    """
    Adaptive preprocessing that improves recognition for ALL skin tones:
      1. CLAHE on LAB L-channel  – lifts shadows / tones down blown highlights
      2. Bilateral filter        – smooths noise, preserves edges
      3. Unsharp mask            – recovers fine edge detail
      4. Gamma correction        – extra boost for very under-exposed frames

    Returns BGR ndarray (same format as input).
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    lab      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b  = cv2.split(lab)
    mean_l   = float(np.mean(l))

    clip  = 3.0 if mean_l < 100 else 2.0
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)

    img_out = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    img_out = cv2.bilateralFilter(img_out, d=9, sigmaColor=75, sigmaSpace=75)

    blur    = cv2.GaussianBlur(img_out, (0, 0), sigmaX=3)
    img_out = cv2.addWeighted(img_out, 1.5, blur, -0.5, 0)

    if mean_l < 80:
        gamma = 1.4
        lut   = np.array([min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                          for i in range(256)], dtype=np.uint8)
        img_out = cv2.LUT(img_out, lut)

    return img_out


def exposure_variants(img_bgr: np.ndarray):
    """Yield original, brightened, darkened versions of a frame."""
    yield img_bgr
    yield cv2.convertScaleAbs(img_bgr, alpha=1.3, beta=20)
    yield cv2.convertScaleAbs(img_bgr, alpha=0.8, beta=-10)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_students() -> dict:
    with open(STUDENTS_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_students(data: dict):
    with open(STUDENTS_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def decode_image(file_storage) -> np.ndarray:
    data = np.frombuffer(file_storage.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def log_attendance(roll: str, name: str, division: str,
                   department: str, status: str = 'present'):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([roll, name, division, department, ts, status])


def already_marked_today(roll: str, status: str) -> bool:
    today = datetime.date.today().isoformat()
    if not os.path.exists(LOG_PATH):
        return False
    with open(LOG_PATH, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if (row.get('Roll No', '').strip() == roll and
                    row.get('Status', '').strip() == status and
                    row.get('Timestamp', '').startswith(today)):
                return True
    return False


def purge_deepface_cache():
    for fname in os.listdir(DB_DIR):
        if fname.endswith('.pkl'):
            try:
                os.remove(os.path.join(DB_DIR, fname))
            except OSError:
                pass


# ── Recognition (multi-strategy + multi-exposure) ────────────────────────────

def smart_recognize(img_bgr: np.ndarray, db_path: str):
    """
    Recognise a face using three exposure variants × two detector backends.
    Returns (roll_key | None, distance | None).
    """
    strategies = [
        {'model_name': DEEPFACE_MODEL, 'detector_backend': DEEPFACE_BACKEND},
        {'model_name': DEEPFACE_MODEL, 'detector_backend': 'opencv'},
    ]

    for variant in exposure_variants(img_bgr):
        processed = preprocess_face(variant)

        for strategy in strategies:
            try:
                results = DeepFace.find(
                    img_path=processed,
                    db_path=db_path,
                    enforce_detection=False,
                    silent=True,
                    distance_metric=DISTANCE_METRIC,
                    **strategy
                )

                if results and len(results[0]) > 0:
                    best     = results[0].iloc[0]
                    dist_col = [c for c in best.index if 'distance' in c.lower()]
                    distance = float(best[dist_col[0]]) if dist_col else 1.0

                    if distance <= CONFIDENCE_THRESHOLD:
                        name = os.path.splitext(
                            os.path.basename(best['identity']))[0]
                        return name, distance

            except Exception:
                continue

    return None, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/register', methods=['POST'])
def register():
    """
    Form fields: image (file), roll, name, division, department.
    Saves face image locally to ./db/<roll>.jpg and updates students.json.
    No cloud storage is used.
    """
    for field in ['roll', 'name', 'division', 'department']:
        if field not in request.form:
            return jsonify({"error": f"Missing field: {field}"}), 400
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    roll       = request.form['roll'].strip().upper()
    name       = request.form['name'].strip()
    division   = request.form['division'].strip().upper()
    department = request.form['department'].strip()

    file       = request.files['image']
    image_data = file.read()

    img_arr = np.frombuffer(image_data, np.uint8)
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # Validate face presence
    detected = False
    for backend in ('retinaface', 'opencv'):
        try:
            faces = DeepFace.extract_faces(
                img_path=preprocess_face(img_bgr),
                enforce_detection=True,
                detector_backend=backend)
            if faces:
                detected = True
                break
        except Exception:
            continue

    if not detected:
        return jsonify({
            "error": "No face detected. Ensure good lighting and face the camera directly."
        }), 400

    # Save preprocessed face image locally
    preprocessed = preprocess_face(img_bgr)
    local_path   = os.path.join(DB_DIR, f"{roll}.jpg")
    cv2.imwrite(local_path, preprocessed)

    students       = load_students()
    students[roll] = {"roll": roll, "name": name,
                      "division": division, "department": department}
    save_students(students)

    purge_deepface_cache()

    return jsonify({
        "status":  "success",
        "message": f"{name} (Roll: {roll}) registered successfully!"
    })


@app.route('/recognize', methods=['POST'])
def recognize():
    """
    Accepts a JPEG image, identifies the face, writes a row to log.csv,
    and returns JSON with student details.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_bgr          = decode_image(request.files['image'])
    name_key, distance = smart_recognize(img_bgr, DB_DIR)

    if name_key is None:
        return jsonify({
            "status":  "unknown",
            "message": "Face not recognised. Please register or improve lighting."
        })

    students   = load_students()
    student    = students.get(name_key, {})
    roll       = student.get('roll',       name_key)
    full_name  = student.get('name',       name_key)
    division   = student.get('division',   'N/A')
    department = student.get('department', 'N/A')
    confidence = round((1 - distance) * 100, 1)

    # Prevent duplicate present entries for the same day
    duplicate = already_marked_today(roll, 'present')
    if not duplicate:
        log_attendance(roll, full_name, division, department, 'present')

    return jsonify({
        "status":     "success",
        "name":       full_name,
        "roll":       roll,
        "division":   division,
        "department": department,
        "confidence": confidence,
        "duplicate":  duplicate,
        "message":    (
            f"Welcome {full_name}! Attendance already marked today."
            if duplicate else
            f"Welcome {full_name}! Attendance marked. ({confidence}% confidence)"
        )
    })


@app.route('/export_csv', methods=['GET'])
def export_csv():
    """Return attendance log as downloadable CSV. ?date=YYYY-MM-DD to filter."""
    date_filter = request.args.get('date', None)
    rows        = []

    with open(LOG_PATH, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if date_filter and len(row) >= 5:
                if not row[4].startswith(date_filter):
                    continue
            rows.append(row)

    buf = io.StringIO()
    w   = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    buf.seek(0)

    filename = f"attendance_{date_filter or datetime.date.today()}.csv"
    return send_file(
        io.BytesIO(buf.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )


@app.route('/students', methods=['GET'])
def list_students():
    return jsonify(load_students())


@app.route('/attendance_today', methods=['GET'])
def attendance_today():
    today   = datetime.date.today().isoformat()
    records = []
    with open(LOG_PATH, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Timestamp', '').startswith(today):
                records.append(row)
    return jsonify(records)


if __name__ == '__main__':
    app.run(debug=True, port=5000)