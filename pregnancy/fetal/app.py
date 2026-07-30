"""
FetalGuard — Flask backend
Two prediction modes:
  1. Image mode      -> EfficientNet-B0 on fetal ultrasound images + Grad-CAM
  2. Numerical mode   -> Random Forest pipeline on maternal vitals

Both modes are explained with a Groq-hosted LLaMA-3 model, using a prompt
that is grounded in the *actual* prediction output for that request (not a
generic template), so the explanation text matches what was really predicted.
"""

import os
import io
import uuid
import traceback
from functools import wraps

import numpy as np
import pandas as pd
import pickle
try:
    import joblib
except ImportError:
    joblib = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

from flask import Flask, request, jsonify, session, send_from_directory, render_template
from werkzeug.security import generate_password_hash, check_password_hash

# Groq client is optional at import time so the app still boots if the
# package or API key isn't configured yet — we fall back to a templated
# explanation in that case instead of crashing.
try:
    from groq import Groq
except ImportError:
    Groq = None


# ============================================================
# App setup
# ============================================================
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

UPLOAD_FOLDER = os.path.join("static", "uploads")
GRADCAM_FOLDER = os.path.join("static", "gradcam")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# MODEL PATHS — edit these two lines to point at your trained checkpoints
# (both can also be overridden with environment variables of the same name)
# ============================================================
IMAGE_MODEL_PATH = os.environ.get(
    "IMAGE_MODEL_PATH",
    r"D:\pregnancy dataset\fetal image\outputs_test\best_model.pth"
)
NUMERICAL_MODEL_PATH = os.environ.get(
    "NUMERICAL_MODEL_PATH",
    r"D:\pregnancy dataset\fetal image\outputs\random_forest_model.pkl"
)


# ============================================================
# In-memory "database" (swap for real DB in production)
# ============================================================
USERS = {}  # email -> {fname, lname, email, role, password_hash}


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "user_email" not in session:
            return jsonify({"ok": False, "error": "Not authenticated"}), 401
        return fn(*args, **kwargs)
    return wrapper


# ============================================================
# Groq client
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY) if (Groq and GROQ_API_KEY) else None
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def generate_groq_explanation(mode, risk_label, confidence, details: dict) -> str:
    """
    Build a prompt grounded in the *actual* prediction for this request and
    ask Groq's LLaMA-3 model to explain it in plain clinical language.
    `details` differs by mode:
      image:      {}  (no structured features, model saw pixels)
      numerical:  {"age":.., "systolic_bp":.., "diastolic_bp":.., ...}
    Falls back to a deterministic templated explanation if Groq isn't
    configured or the call fails, so the UI never shows a blank/wrong note.
    """
    if mode == "numerical":
        feature_lines = "\n".join(f"- {k}: {v}" for k, v in details.items())
        prompt = (
            "You are a clinical assistant explaining an AI pregnancy risk "
            "screening result to an obstetric care team. The prediction was "
            f"produced by a machine learning model trained on maternal vitals "
            "and history.\n\n"
            f"Predicted risk category: {risk_label}\n"
            f"Model confidence: {confidence}%\n"
            f"Patient input values:\n{feature_lines}\n\n"
            "In 3-4 sentences, explain in plain clinical language which of "
            "the input values are most likely driving this risk category "
            "(reference normal clinical ranges where relevant), and note any "
            "values that look borderline. Do not invent findings that are "
            "not supported by the values given. Do not give a diagnosis — "
            "frame this as a screening signal for clinician review."
        )
    else:  # image mode
        prompt = (
            "You are a clinical assistant explaining an AI pregnancy risk "
            "screening result to an obstetric care team. A deep learning "
            "model (EfficientNet-B0) analyzed a fetal ultrasound image and "
            "produced the following classification. A Grad-CAM heat map was "
            "also generated, highlighting the image regions that most "
            "influenced the model's decision.\n\n"
            f"Predicted risk category: {risk_label}\n"
            f"Model confidence: {confidence}%\n\n"
            "In 3-4 sentences, explain in plain clinical language what this "
            "risk category generally means for antenatal follow-up, and "
            "describe in general terms what a Grad-CAM heat map is showing "
            "(i.e. the regions of the ultrasound the model focused on) "
            "without inventing specific anatomical findings you cannot see. "
            "Frame this as a screening signal for clinician review, not a "
            "diagnosis."
        )

    if groq_client is None:
        return _fallback_explanation(mode, risk_label, details)

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful, concise clinical AI assistant. You never state a diagnosis, only screening-level observations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        text = completion.choices[0].message.content.strip()
        return text if text else _fallback_explanation(mode, risk_label, details)
    except Exception as e:
        print("Groq call failed:", e)
        return _fallback_explanation(mode, risk_label, details)


def _fallback_explanation(mode, risk_label, details: dict) -> str:
    """Deterministic backup explanation used if Groq is unavailable."""
    if mode == "numerical":
        flags = []
        if "systolic_bp" in details and details["systolic_bp"] >= 140:
            flags.append("elevated systolic blood pressure")
        if "diastolic_bp" in details and details["diastolic_bp"] >= 90:
            flags.append("elevated diastolic blood pressure")
        if "blood_sugar" in details and details["blood_sugar"] >= 7.8:
            flags.append("elevated blood glucose")
        if "body_temp" in details and details["body_temp"] >= 100.4:
            flags.append("elevated body temperature")
        if "heart_rate" in details and details["heart_rate"] >= 100:
            flags.append("elevated heart rate")
        if "bmi" in details and details["bmi"] >= 30:
            flags.append("elevated BMI")
        if details.get("previous_complications"):
            flags.append("history of previous pregnancy complications")
        if details.get("preexisting_diabetes"):
            flags.append("preexisting diabetes")
        if details.get("gestational_diabetes"):
            flags.append("gestational diabetes")
        if details.get("mental_health"):
            flags.append("reported mental health concerns")
        if flags:
            flag_text = ", ".join(flags)
            return (f"The model classified this case as {risk_label} based on the submitted vitals and history. "
                    f"Notable contributing factors include {flag_text}, which fall outside typical "
                    f"reference ranges or add to cumulative risk. This is a screening signal only — please correlate "
                    f"with a full clinical assessment.")
        return (f"The model classified this case as {risk_label} based on the submitted vitals and history, which "
                f"fall largely within typical reference ranges. This is a screening signal only — please "
                f"correlate with a full clinical assessment.")
    return (f"The model classified this ultrasound image as {risk_label}. The Grad-CAM heat map highlights "
            f"the regions of the image that most influenced this classification. This is a screening signal "
            f"only and should be correlated with a full clinical assessment.")


# ============================================================
# Advice text (mirrors the front-end ADVICE map, used as fallback / API source)
# ============================================================
ADVICE = {
    "High Risk": "Escalate for urgent maternal-fetal medicine review.",
    "Medium Risk": "Increase monitoring frequency and schedule a follow-up soon.",
    "Low Risk": "Continue routine antenatal care.",
}


# ============================================================
# IMAGE MODEL — EfficientNet-B0
# (IMAGE_MODEL_PATH is set in the MODEL PATHS config block near the top)
# ============================================================
image_model = None
image_classes = None
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image_model():
    global image_model, image_classes
    if not os.path.exists(IMAGE_MODEL_PATH):
        print(f"[image model] WARNING: not found at {IMAGE_MODEL_PATH} — image predictions will fail until you set IMAGE_MODEL_PATH")
        return
    checkpoint = torch.load(IMAGE_MODEL_PATH, map_location=device)
    image_classes = checkpoint["classes"]

    m = efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, len(image_classes)))
    m.load_state_dict(checkpoint["model_state_dict"])
    m.to(device)
    m.eval()

    image_model = m
    print("[image model] loaded. Classes:", image_classes)


# ------------------------------------------------------------
# Grad-CAM for EfficientNet-B0
# ------------------------------------------------------------
class GradCAM:
    """Grad-CAM hooked onto the last convolutional block of EfficientNet-B0."""

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.features[-1]  # last conv block
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients[0]          # (C, H, W)
        activations = self.activations[0]       # (C, H, W)
        weights = gradients.mean(dim=(1, 2))     # (C,)

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()


def make_gradcam_overlay(original_pil_img, cam, out_path):
    """Resize CAM to original image size, colorize, and blend as an overlay."""
    import matplotlib.cm as cm

    w, h = original_pil_img.size
    cam_resized = Image.fromarray(np.uint8(cam * 255)).resize((w, h), resample=Image.BILINEAR)
    cam_arr = np.array(cam_resized) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap = colormap(cam_arr)[:, :, :3]  # drop alpha
    heatmap = np.uint8(heatmap * 255)
    heatmap_img = Image.fromarray(heatmap).convert("RGB")

    base = original_pil_img.convert("RGB")
    overlay = Image.blend(base, heatmap_img, alpha=0.45)
    overlay.save(out_path)


# ============================================================
# IMAGE-MODE RISK LABELS
#
# The image classifier is a 3-class model over {Low, Medium, High} Risk.
# This is intentionally kept separate from normalize_risk_label() below,
# which is used by the *numerical* model and is calibrated for that
# model's confirmed binary classes_ == [0, 1]. Reusing the binary map for
# the image model would either mis-map a 3rd class or fall through to an
# invented label like "Class 2" that isn't one of the three categories
# the UI/ADVICE map understands.
# ============================================================

# The only labels this endpoint is allowed to return.
IMAGE_RISK_LABELS = ("Low Risk", "Medium Risk", "High Risk")

# If image_classes (checkpoint["classes"]) turns out to be numeric-encoded
# (0/1/2) rather than strings, this defines what each index means. Check
# the "[image model] loaded. Classes: ..." log line at startup and edit
# this mapping if your training script used a different order.
IMAGE_NUMERIC_LABEL_MAP = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def normalize_image_risk_label(raw_label) -> str:
    """Strictly map the image model's class output onto one of the three
    canonical risk categories (Low/Medium/High Risk). Unlike
    normalize_risk_label() (used by the binary numerical model), this
    never falls back to an invented 'Class N' or arbitrary title-cased
    string — if the label can't be confidently mapped, it logs a warning
    and defaults to Medium Risk so an unrecognized class doesn't silently
    read as reassuring (Low) or alarming (High) when we're not sure."""
    if isinstance(raw_label, (int, np.integer)):
        label = IMAGE_NUMERIC_LABEL_MAP.get(int(raw_label))
        if label:
            return label
        print(f"[image model] WARNING: unrecognized numeric class {raw_label}, defaulting to Medium Risk")
        return "Medium Risk"

    s = str(raw_label).strip().lower()
    if s.isdigit():
        label = IMAGE_NUMERIC_LABEL_MAP.get(int(s))
        if label:
            return label
        print(f"[image model] WARNING: unrecognized numeric class '{s}', defaulting to Medium Risk")
        return "Medium Risk"

    if "high" in s:
        return "High Risk"
    if "mid" in s or "medium" in s:
        return "Medium Risk"
    if "low" in s:
        return "Low Risk"

    print(f"[image model] WARNING: unrecognized class label '{raw_label}', defaulting to Medium Risk")
    return "Medium Risk"


# ============================================================
# NUMERICAL MODEL — scikit-learn Pipeline (preprocessor + Random Forest)
# on maternal vitals + history.
#
# IMPORTANT — this was the source of the "columns are missing" error:
# the actual random_forest_model.pkl on disk was NOT trained on the
# 6-field {Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate} schema
# this file originally assumed. Inspecting the pipeline's
# `feature_names_in_` shows it actually expects 11 columns, exactly in
# this order:
#
#   ['Age', 'Systolic BP', 'Diastolic', 'BS', 'Body Temp', 'BMI',
#    'Previous Complications', 'Preexisting Diabetes',
#    'Gestational Diabetes', 'Mental Health', 'Heart Rate']
#
# The last five (Previous Complications, Preexisting Diabetes,
# Gestational Diabetes, Mental Health, and BMI) were missing from the
# original form/API entirely. CSV_FEATURE_COLUMNS/NUMERICAL_FEATURES
# below have been corrected to match, and the front-end form/JS payload
# need the extra fields too (see index.html).
#
# ALSO IMPORTANT: this pipeline's classifier.classes_ is [0, 1] — it's a
# BINARY classifier (not 3-class Low/Medium/High like the rest of the
# UI assumes). There is no target_encoder.pkl bundled with this
# deployment to tell us what 0 and 1 mean semantically, so
# NUMERIC_LABEL_MAP below assumes the common convention 0 = Low Risk,
# 1 = High Risk. If your training script used the opposite convention,
# flip the map in normalize_risk_label() below.
# ============================================================

# Exact column names/order the pipeline's ColumnTransformer was fit on
# (from pipeline.feature_names_in_).
CSV_FEATURE_COLUMNS = [
    "Age", "Systolic BP", "Diastolic", "BS", "Body Temp", "BMI",
    "Previous Complications", "Preexisting Diabetes",
    "Gestational Diabetes", "Mental Health", "Heart Rate",
]

# Keys the front end sends in the JSON body, mapped 1:1 (same order) to
# CSV_FEATURE_COLUMNS above.
NUMERICAL_FEATURES = [
    "age", "systolic_bp", "diastolic_bp", "blood_sugar", "body_temp", "bmi",
    "previous_complications", "preexisting_diabetes",
    "gestational_diabetes", "mental_health", "heart_rate",
]

# These are sent from the front end as 0/1 (No/Yes) rather than free numbers.
BINARY_FEATURES = {
    "previous_complications", "preexisting_diabetes",
    "gestational_diabetes", "mental_health",
}

# target_encoder.pkl lives alongside random_forest_model.pkl by default;
# override with the TARGET_ENCODER_PATH env var if it's stored elsewhere.
TARGET_ENCODER_PATH = os.environ.get(
    "TARGET_ENCODER_PATH",
    os.path.join(os.path.dirname(NUMERICAL_MODEL_PATH), "target_encoder.pkl")
)

numerical_model = None    # the fitted sklearn Pipeline (preprocessor + classifier)
target_encoder = None     # the fitted sklearn LabelEncoder used on the training target, if present


def _load_pickle(path):
    try:
        if joblib:
            return joblib.load(path)
    except Exception:
        pass
    with open(path, "rb") as f:
        return pickle.load(f)


def load_numerical_model():
    global numerical_model, target_encoder

    if not os.path.exists(NUMERICAL_MODEL_PATH):
        print(f"[numerical model] WARNING: not found at {NUMERICAL_MODEL_PATH} — numerical predictions will fail until you set NUMERICAL_MODEL_PATH")
        return

    numerical_model = _load_pickle(NUMERICAL_MODEL_PATH)
    print(f"[numerical model] loaded pipeline from {NUMERICAL_MODEL_PATH}")
    if hasattr(numerical_model, "feature_names_in_"):
        print("[numerical model] expects columns:", list(numerical_model.feature_names_in_))

    if not os.path.exists(TARGET_ENCODER_PATH):
        print(f"[numerical model] NOTE: target_encoder.pkl not found at {TARGET_ENCODER_PATH} — "
              f"falling back to the classifier's own classes_ + NUMERIC_LABEL_MAP for labels.")
        target_encoder = None
        return

    target_encoder = _load_pickle(TARGET_ENCODER_PATH)
    print("[numerical model] loaded target_encoder. Classes:", list(getattr(target_encoder, "classes_", [])))


def normalize_risk_label(raw_label) -> str:
    """Map arbitrary dataset class labels onto the front-end's expected
    labels. Handles string labels (e.g. 'low risk', 'high risk') as well
    as numeric-encoded labels. This deployment's model is BINARY
    (classes_ == [0, 1]), so NUMERIC_LABEL_MAP only has two entries —
    edit it if 0/1 mean the opposite of what's assumed here."""
    NUMERIC_LABEL_MAP = {0: "Low Risk", 1: "High Risk"}

    if isinstance(raw_label, (int, np.integer)):
        return NUMERIC_LABEL_MAP.get(int(raw_label), f"Class {raw_label}")

    s = str(raw_label).strip().lower()
    if s.isdigit():
        return NUMERIC_LABEL_MAP.get(int(s), f"Class {s}")
    if "high" in s:
        return "High Risk"
    if "mid" in s or "medium" in s:
        return "Medium Risk"
    if "low" in s:
        return "Low Risk"
    return str(raw_label).title()


# ============================================================
# Auth routes
# ============================================================
@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or {}
    fname = (data.get("fname") or "").strip()
    lname = (data.get("lname") or "").strip()
    email = (data.get("email") or "").strip().lower()
    role = (data.get("role") or "").strip()
    password = data.get("password") or ""

    if not fname or not email or not password:
        return jsonify({"ok": False, "error": "Please fill in all required fields."}), 400
    if len(password) < 8:
        return jsonify({"ok": False, "error": "Password must be at least 8 characters."}), 400
    if email in USERS:
        return jsonify({"ok": False, "error": "An account with this email already exists."}), 400

    USERS[email] = {
        "fname": fname,
        "lname": lname,
        "email": email,
        "role": role,
        "password_hash": generate_password_hash(password),
    }
    session["user_email"] = email
    return jsonify({"ok": True})


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = USERS.get(email)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"ok": False, "error": "Invalid email or password."}), 401

    session["user_email"] = email
    return jsonify({"ok": True})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/session", methods=["GET"])
def api_session():
    email = session.get("user_email")
    if not email or email not in USERS:
        return jsonify({"ok": False}), 401
    u = USERS[email]
    return jsonify({"ok": True, "user": {"fname": u["fname"], "lname": u["lname"], "email": u["email"], "role": u["role"]}})


# ============================================================
# Prediction routes
# ============================================================
@app.route("/api/predict/image", methods=["POST"])
@login_required
def api_predict_image():
    if image_model is None:
        return jsonify({"ok": False, "error": "Image model is not loaded on the server. Set IMAGE_MODEL_PATH."}), 503

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"ok": False, "error": "No image file provided."}), 400

    file = request.files["image"]
    try:
        unique_name = f"{uuid.uuid4().hex}_{file.filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(save_path)

        pil_img = Image.open(save_path).convert("RGB")
        input_tensor = image_transform(pil_img).unsqueeze(0).to(device)

        # Standard forward pass for prediction + confidence
        with torch.no_grad():
            output = image_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence_value, predicted_idx = torch.max(probs, dim=1)

        predicted_idx = predicted_idx.item()
        raw_label = image_classes[predicted_idx]
        risk_label = normalize_image_risk_label(raw_label)
        confidence = round(confidence_value.item() * 100, 2)

        # Grad-CAM needs gradients, so run a second pass with grad enabled
        cam_engine = GradCAM(image_model)
        input_tensor_grad = image_transform(pil_img).unsqueeze(0).to(device)
        input_tensor_grad.requires_grad_(True)
        cam = cam_engine.generate(input_tensor_grad, predicted_idx)

        gradcam_name = f"gradcam_{unique_name}.png"
        gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_name)
        make_gradcam_overlay(pil_img, cam, gradcam_path)
        gradcam_url = f"/static/gradcam/{gradcam_name}"

        explanation = generate_groq_explanation("image", risk_label, confidence, {})

        return jsonify({
            "ok": True,
            "prediction": risk_label,
            "confidence": confidence,
            "advice": ADVICE.get(risk_label, ""),
            "gradcam_url": gradcam_url,
            "groq_explanation": explanation,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Prediction failed: {e}"}), 500


@app.route("/api/predict/numerical", methods=["POST"])
@login_required
def api_predict_numerical():
    if numerical_model is None:
        return jsonify({"ok": False, "error": "Numerical model is not loaded on the server. Set NUMERICAL_MODEL_PATH."}), 503

    data = request.get_json(silent=True) or {}

    try:
        values = []
        details = {}
        for feat in NUMERICAL_FEATURES:
            if feat not in data or data[feat] in (None, ""):
                return jsonify({"ok": False, "error": f"Missing value for '{feat}'."}), 400
            if feat in BINARY_FEATURES:
                # Accept booleans, "yes"/"no" strings, or 0/1 numbers from the client.
                raw = data[feat]
                if isinstance(raw, bool):
                    val = 1.0 if raw else 0.0
                elif isinstance(raw, str):
                    val = 1.0 if raw.strip().lower() in ("1", "yes", "true") else 0.0
                else:
                    val = float(raw)
            else:
                val = float(data[feat])
            values.append(val)
            details[feat] = val
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "All fields must be numeric."}), 400

    try:
        # The pipeline's ColumnTransformer was fit on a DataFrame with the
        # original CSV column names — it needs the same here. Preprocessing
        # (imputation + scaling) happens inside the pipeline itself, so we
        # pass raw values, in the same order as NUMERICAL_FEATURES /
        # CSV_FEATURE_COLUMNS.
        input_df = pd.DataFrame([values], columns=CSV_FEATURE_COLUMNS)

        probs = numerical_model.predict_proba(input_df)[0]
        predicted_idx = int(np.argmax(probs))
        confidence = round(float(probs[predicted_idx]) * 100, 2)

        if target_encoder is not None:
            raw_label = target_encoder.inverse_transform([predicted_idx])[0]
        else:
            # No encoder available — fall back to whatever the pipeline's
            # classifier itself reports for classes_ (may be raw integers).
            classifier = numerical_model.named_steps.get("classifier", numerical_model) \
                if hasattr(numerical_model, "named_steps") else numerical_model
            classes = list(getattr(classifier, "classes_", []))
            raw_label = classes[predicted_idx] if predicted_idx < len(classes) else predicted_idx

        risk_label = normalize_risk_label(raw_label)

        explanation = generate_groq_explanation("numerical", risk_label, confidence, details)

        return jsonify({
            "ok": True,
            "prediction": risk_label,
            "confidence": confidence,
            "advice": ADVICE.get(risk_label, ""),
            "groq_explanation": explanation,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Prediction failed: {e}"}), 500


# ============================================================
# Static file serving for gradcam overlays
# ============================================================
@app.route("/static/gradcam/<path:filename>")
def serve_gradcam(filename):
    return send_from_directory(GRADCAM_FOLDER, filename)


# ============================================================
# Front-end
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


# ============================================================
# Boot
# ============================================================
if __name__ == "__main__":
    load_image_model()
    load_numerical_model()
    print("Starting Flask Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
