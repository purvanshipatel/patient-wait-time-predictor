# ============================================================
# PATIENT QUEUE WAIT TIME PREDICTOR - FULL STACK BACKEND
# FLASK + POSTGRESQL (RENDER) + SCIKIT-LEARN
# ============================================================
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# ------------------------------------------------------------
# 1. DATABASE CONFIGURATION (POSTGRESQL ON RENDER / SQLITE LOCAL)
# ------------------------------------------------------------
database_url = os.environ.get("DATABASE_URL")
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = database_url or "sqlite:///ed_queue.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ------------------------------------------------------------
# 2. DATABASE MODEL: PATIENT PREDICTION LOG
# ------------------------------------------------------------
class PatientQueueRecord(db.Model):
    __tablename__ = "patient_queue_records"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Input features
    triage_level = db.Column(db.Integer, nullable=False)
    chief_complaint = db.Column(db.String(120), nullable=False)
    department = db.Column(db.String(80), nullable=False)
    attending_physician = db.Column(db.String(120), nullable=False)
    patient_age = db.Column(db.Integer, nullable=False)
    patient_gender = db.Column(db.String(20), nullable=False)
    insurance_type = db.Column(db.String(80), nullable=False)
    length_of_stay_hrs = db.Column(db.Float, nullable=False)
    disposition = db.Column(db.String(80), nullable=False)
    readmitted_30d = db.Column(db.Integer, nullable=False)
    satisfaction_score = db.Column(db.Float, nullable=False)
    arrival_hour = db.Column(db.Integer, nullable=False)
    arrival_dayofweek = db.Column(db.Integer, nullable=False)
    arrival_month = db.Column(db.Integer, nullable=False)
    age_group = db.Column(db.String(40), nullable=False)

    # Prediction outcome
    predicted_wait_min = db.Column(db.Float, nullable=False)
    queue_status = db.Column(db.String(30), default="Waiting")

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "triage_level": self.triage_level,
            "chief_complaint": self.chief_complaint,
            "department": self.department,
            "attending_physician": self.attending_physician,
            "patient_age": self.patient_age,
            "patient_gender": self.patient_gender,
            "predicted_wait_min": round(self.predicted_wait_min, 1),
            "queue_status": self.queue_status
        }

# ------------------------------------------------------------
# 3. LOAD SAVED MODEL AND METADATA
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "best_wait_time_model.pkl")
METADATA_FILE = os.path.join(BASE_DIR, "project_metadata.pkl")

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

model = joblib.load(MODEL_FILE)
metadata = joblib.load(METADATA_FILE)

FEATURE_COLUMNS = metadata["feature_columns"]
TARGET_COLUMN = metadata["target_column"]
MODEL_NAME = metadata["model_name"]

# Create database tables automatically upon startup
with app.app_context():
    db.create_all()

# ------------------------------------------------------------
# 4. ROUTES
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/features", methods=["GET"])
def get_features():
    return jsonify({
        "required_features": FEATURE_COLUMNS,
        "total_features": len(FEATURE_COLUMNS),
        "prediction_target": TARGET_COLUMN,
        "model_name": MODEL_NAME
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        patient_data = request.get_json(silent=True)
        if not isinstance(patient_data, dict):
            return jsonify({"error": "Send patient data as a JSON object."}), 400

        missing_features = [col for col in FEATURE_COLUMNS if col not in patient_data]
        if missing_features:
            return jsonify({
                "error": "Some required features are missing.",
                "missing_features": missing_features
            }), 400

        # Construct dataframe matching training feature order
        input_df = pd.DataFrame(
            [[patient_data[col] for col in FEATURE_COLUMNS]],
            columns=FEATURE_COLUMNS
        )

        # Model prediction
        prediction = float(model.predict(input_df)[0])
        if not np.isfinite(prediction) or prediction < 0:
            return jsonify({"error": "The model returned an invalid prediction."}), 500

        rounded_wait = round(prediction, 2)

        # Save record into PostgreSQL
        record = PatientQueueRecord(
            triage_level=int(patient_data["triage_level"]),
            chief_complaint=str(patient_data["chief_complaint"]),
            department=str(patient_data["department"]),
            attending_physician=str(patient_data["attending_physician"]),
            patient_age=int(patient_data["patient_age"]),
            patient_gender=str(patient_data["patient_gender"]),
            insurance_type=str(patient_data["insurance_type"]),
            length_of_stay_hrs=float(patient_data["length_of_stay_hrs"]),
            disposition=str(patient_data["disposition"]),
            readmitted_30d=int(patient_data["readmitted_30d"]),
            satisfaction_score=float(patient_data["satisfaction_score"]),
            arrival_hour=int(patient_data["arrival_hour"]),
            arrival_dayofweek=int(patient_data["arrival_dayofweek"]),
            arrival_month=int(patient_data["arrival_month"]),
            age_group=str(patient_data["age_group"]),
            predicted_wait_min=rounded_wait,
            queue_status="Waiting"
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            "record_id": record.id,
            "model": MODEL_NAME,
            "predicted_waiting_time_minutes": rounded_wait,
            "unit": "minutes",
            "saved_to_db": True
        })

    except Exception as error:
        db.session.rollback()
        app.logger.exception("Prediction or DB save failed")
        return jsonify({"error": "Failed to process prediction.", "details": str(error)}), 500

@app.route("/api/queue", methods=["GET"])
def get_queue():
    """Fetch the latest 50 patient records from PostgreSQL."""
    records = PatientQueueRecord.query.order_by(PatientQueueRecord.id.desc()).limit(50).all()
    return jsonify([r.to_dict() for r in records])

@app.route("/api/queue/<int:record_id>/status", methods=["PATCH"])
def update_status(record_id):
    """Update patient queue status (e.g. Waiting -> With Doctor -> Completed)."""
    data = request.get_json(silent=True) or {}
    new_status = data.get("status")
    if new_status not in ["Waiting", "With Doctor", "Completed"]:
        return jsonify({"error": "Invalid status value."}), 400

    record = PatientQueueRecord.query.get_or_404(record_id)
    record.queue_status = new_status
    db.session.commit()
    return jsonify({"success": True, "record": record.to_dict()})

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Aggregate statistics from database."""
    total_patients = PatientQueueRecord.query.count()
    waiting_count = PatientQueueRecord.query.filter_by(queue_status="Waiting").count()
    
    avg_wait = db.session.query(db.func.avg(PatientQueueRecord.predicted_wait_min)).scalar() or 0.0
    
    return jsonify({
        "total_intake": total_patients,
        "current_waiting": waiting_count,
        "avg_wait_min": round(float(avg_wait), 1)
    })

# ------------------------------------------------------------
# 5. SERVER ENTRY
# ------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
