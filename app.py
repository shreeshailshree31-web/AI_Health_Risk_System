# ============================================================
# AI HEALTH RISK SYSTEM
# Flask + MongoDB + Machine Learning
# ============================================================

# ------------------------------------------------------------
# IMPORTANT:
# Limit numerical-library threads BEFORE importing numpy/sklearn
# ------------------------------------------------------------
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

from datetime import datetime
import joblib


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "ai-health-system-secret-key-change-later"
)


# ============================================================
# MONGODB CONNECTION
# ============================================================

client = None
db = None
users_collection = None
assessments_collection = None

try:

    if not MONGO_URI:
        raise ValueError("MONGO_URI not found in .env file")

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000
    )

    # Test connection
    client.admin.command("ping")

    db = client["healthcare_db"]

    users_collection = db["users"]
    assessments_collection = db["assessments"]

    print("\n========================================")
    print(" MongoDB CONNECTION SUCCESSFUL")
    print("========================================\n")

except Exception as e:

    print("\n========================================")
    print(" MONGODB CONNECTION FAILED")
    print("========================================")
    print("Error:", e)
    print("========================================\n")


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "health_risk_model.pkl"
)

try:

    model = joblib.load(MODEL_PATH)

    print("========================================")
    print(" AI MODEL LOADED SUCCESSFULLY")
    print("========================================")
    print()

except Exception as e:

    model = None

    print("\n========================================")
    print(" AI MODEL LOAD FAILED")
    print("========================================")
    print("Error:", e)
    print("========================================\n")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # Basic validation
        if not name or not email or not password:

            flash("Please fill in all fields.", "error")

            return redirect(url_for("register"))

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "error"
            )

            return redirect(url_for("register"))

        # Check MongoDB
        if users_collection is None:

            flash(
                "Database connection is unavailable.",
                "error"
            )

            return redirect(url_for("register"))

        # Check existing account
        existing_user = users_collection.find_one(
            {"email": email}
        )

        if existing_user:

            flash(
                "An account with this email already exists.",
                "error"
            )

            return redirect(url_for("login"))

        # Hash password
        hashed_password = generate_password_hash(password)

        user = {
            "name": name,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.utcnow()
        }

        users_collection.insert_one(user)

        flash(
            "Account created successfully. Please login.",
            "success"
        )

        return redirect(url_for("login"))

    return render_template("register.html")


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:

            flash(
                "Please enter email and password.",
                "error"
            )

            return redirect(url_for("login"))

        if users_collection is None:

            flash(
                "Database connection is unavailable.",
                "error"
            )

            return redirect(url_for("login"))

        user = users_collection.find_one(
            {"email": email}
        )

        if user and check_password_hash(
            user["password"],
            password
        ):

            session["user_id"] = str(user["_id"])
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]

            flash(
                "Login successful!",
                "success"
            )

            return redirect(url_for("dashboard"))

        flash(
            "Invalid email or password.",
            "error"
        )

        return redirect(url_for("login"))

    return render_template("login.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(url_for("home"))


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        flash(
            "Please login first.",
            "error"
        )

        return redirect(url_for("login"))

    recent_assessment = None

    if assessments_collection is not None:

        recent_assessment = assessments_collection.find_one(
            {"user_id": session["user_id"]},
            sort=[("created_at", -1)]
        )

    return render_template(
        "dashboard.html",
        user_name=session.get("user_name"),
        recent_assessment=recent_assessment
    )


# ============================================================
# HEALTH ASSESSMENT
# ============================================================

@app.route("/assessment", methods=["GET", "POST"])
def assessment():

    if "user_id" not in session:

        flash(
            "Please login to complete your assessment.",
            "error"
        )

        return redirect(url_for("login"))

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # GET FORM DATA
            # ------------------------------------------------

            age = float(
                request.form.get("age", 0)
            )

            height = float(
                request.form.get("height", 0)
            )

            weight = float(
                request.form.get("weight", 0)
            )

            systolic_bp = float(
                request.form.get("systolic_bp", 0)
            )

            glucose = float(
                request.form.get("glucose", 0)
            )

            cholesterol = float(
                request.form.get("cholesterol", 0)
            )

            smoking = int(
                request.form.get("smoking", 0)
            )

            physical_activity = int(
                request.form.get("physical_activity", 0)
            )

            family_history = int(
                request.form.get("family_history", 0)
            )

            sleep_hours = float(
                request.form.get("sleep_hours", 0)
            )


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if height <= 0 or weight <= 0:

                flash(
                    "Please enter valid height and weight.",
                    "error"
                )

                return redirect(url_for("assessment"))

            if age <= 0:

                flash(
                    "Please enter a valid age.",
                    "error"
                )

                return redirect(url_for("assessment"))


            # ------------------------------------------------
            # BMI CALCULATION
            # ------------------------------------------------

            height_meters = height / 100

            bmi = weight / (
                height_meters ** 2
            )

            bmi = round(bmi, 2)


            # ------------------------------------------------
            # SLEEP RISK
            # ------------------------------------------------

            if sleep_hours < 6:

                poor_sleep = 1

            else:

                poor_sleep = 0


            # ------------------------------------------------
            # MACHINE LEARNING FEATURES
            #
            # IMPORTANT:
            # This order MUST match train_model.py
            # ------------------------------------------------

            features = [[

                age,
                bmi,
                systolic_bp,
                glucose,
                cholesterol,
                smoking,
                physical_activity,
                family_history,
                poor_sleep

            ]]


            # ------------------------------------------------
            # AI PREDICTION
            # ------------------------------------------------

            if model is None:

                flash(
                    "AI model is unavailable.",
                    "error"
                )

                return redirect(
                    url_for("assessment")
                )

            prediction = model.predict(
                features
            )[0]

            # Probability
            try:

                probabilities = model.predict_proba(
                    features
                )[0]

                confidence = max(
                    probabilities
                ) * 100

                confidence = round(
                    confidence,
                    1
                )

            except Exception:

                confidence = None


            # ------------------------------------------------
            # PERSONALIZED RECOMMENDATIONS
            # ------------------------------------------------

            recommendations = []


            if bmi >= 30:

                recommendations.append(
                    "Focus on gradual weight management through a balanced diet and regular physical activity."
                )

            elif bmi < 18.5:

                recommendations.append(
                    "Consider maintaining a balanced nutrient-rich diet and discuss healthy weight goals with a professional."
                )

            else:

                recommendations.append(
                    "Your BMI is within the commonly used healthy range. Continue maintaining balanced nutrition and activity."
                )


            if systolic_bp >= 140:

                recommendations.append(
                    "Your systolic blood pressure reading is elevated. Consider monitoring it regularly and discussing it with a healthcare professional."
                )

            else:

                recommendations.append(
                    "Continue monitoring blood pressure as part of regular preventive healthcare."
                )


            if glucose >= 126:

                recommendations.append(
                    "Your glucose value is elevated. Consider discussing the result with a qualified healthcare professional."
                )


            if cholesterol >= 240:

                recommendations.append(
                    "Your cholesterol value is elevated. A heart-healthy diet and professional evaluation may be beneficial."
                )


            if smoking == 1:

                recommendations.append(
                    "Avoiding tobacco products can significantly support long-term health."
                )


            if physical_activity == 0:

                recommendations.append(
                    "Try to include regular physical activity in your routine, according to your abilities."
                )


            if poor_sleep == 1:

                recommendations.append(
                    "Aim for a consistent sleep routine and adequate sleep duration."
                )


            if family_history == 1:

                recommendations.append(
                    "Because family history can influence health risk, regular preventive check-ups can be useful."
                )


            # ------------------------------------------------
            # SAVE ASSESSMENT TO MONGODB
            # ------------------------------------------------

            assessment_document = {

                "user_id": session["user_id"],

                "age": age,

                "height": height,

                "weight": weight,

                "bmi": bmi,

                "systolic_bp": systolic_bp,

                "glucose": glucose,

                "cholesterol": cholesterol,

                "smoking": smoking,

                "physical_activity": physical_activity,

                "family_history": family_history,

                "sleep_hours": sleep_hours,

                "poor_sleep": poor_sleep,

                "risk": prediction,

                "confidence": confidence,

                "recommendations": recommendations,

                "created_at": datetime.utcnow()

            }


            if assessments_collection is not None:

                assessments_collection.insert_one(
                    assessment_document
                )


            # ------------------------------------------------
            # RESULT PAGE
            # ------------------------------------------------

            return render_template(
                "result.html",

                risk=prediction,

                confidence=confidence,

                bmi=bmi,

                recommendations=recommendations,

                assessment=assessment_document
            )


        except ValueError:

            flash(
                "Please enter valid numbers in all health fields.",
                "error"
            )

            return redirect(
                url_for("assessment")
            )

        except Exception as e:

            print(
                "\nAssessment Error:",
                e
            )

            flash(
                "Something went wrong while processing the assessment.",
                "error"
            )

            return redirect(
                url_for("assessment")
            )


    return render_template(
        "assessment.html"
    )


# ============================================================
# ASSESSMENT HISTORY
# ============================================================

@app.route("/history")
def history():

    if "user_id" not in session:

        flash(
            "Please login first.",
            "error"
        )

        return redirect(
            url_for("login")
        )

    assessments = []

    if assessments_collection is not None:

        assessments = list(
            assessments_collection.find(
                {
                    "user_id": session["user_id"]
                }
            ).sort(
                "created_at",
                -1
            )
        )

    return render_template(
        "history.html",
        assessments=assessments
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "index.html"
    ), 404


@app.errorhandler(500)
def internal_server_error(error):

    return """
    <h1>Something went wrong</h1>
    <p>Please return to the HealthAI homepage and try again.</p>
    """, 500


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("==============================================")
    print("       AI HEALTH RISK SYSTEM")
    print("==============================================")
    print("Server: http://127.0.0.1:5000")
    print("Open the above address in your browser.")
    print("==============================================")
    print("\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )