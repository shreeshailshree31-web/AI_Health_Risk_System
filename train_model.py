import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# ============================================================
# AI HEALTH RISK PREDICTION MODEL
# ============================================================

print("\n======================================")
print(" AI HEALTH RISK PREDICTION MODEL")
print("======================================\n")


# ------------------------------------------------------------
# 1. Generate synthetic health data
# ------------------------------------------------------------

np.random.seed(42)

n_samples = 1000

age = np.random.randint(18, 80, n_samples)

bmi = np.round(
    np.random.uniform(17, 40, n_samples),
    1
)

systolic_bp = np.random.randint(
    90,
    181,
    n_samples
)

glucose = np.random.randint(
    70,
    201,
    n_samples
)

cholesterol = np.random.randint(
    120,
    281,
    n_samples
)

smoking = np.random.randint(
    0,
    2,
    n_samples
)

physical_activity = np.random.randint(
    0,
    2,
    n_samples
)

family_history = np.random.randint(
    0,
    2,
    n_samples
)

poor_sleep = np.random.randint(
    0,
    2,
    n_samples
)


# ------------------------------------------------------------
# 2. Calculate a preventive health risk score
# ------------------------------------------------------------

risk_score = (

    (age > 50) * 1

    + (bmi >= 30) * 2

    + (systolic_bp >= 140) * 2

    + (glucose >= 126) * 2

    + (cholesterol >= 240) * 2

    + smoking * 1

    + (physical_activity == 0) * 1

    + family_history * 1

    + poor_sleep * 1
)


# ------------------------------------------------------------
# 3. Convert score into risk categories
# ------------------------------------------------------------

def classify_risk(score):

    if score <= 2:
        return "Low"

    elif score <= 5:
        return "Moderate"

    else:
        return "High"


risk = np.array([
    classify_risk(score)
    for score in risk_score
])


# ------------------------------------------------------------
# 4. Create Pandas DataFrame
# ------------------------------------------------------------

data = pd.DataFrame({

    "age": age,

    "bmi": bmi,

    "systolic_bp": systolic_bp,

    "glucose": glucose,

    "cholesterol": cholesterol,

    "smoking": smoking,

    "physical_activity": physical_activity,

    "family_history": family_history,

    "poor_sleep": poor_sleep,

    "risk": risk

})


print("Dataset created successfully.")

print(f"Number of records: {len(data)}")

print("\nRisk distribution:")

print(data["risk"].value_counts())


# ------------------------------------------------------------
# 5. Separate features and target
# ------------------------------------------------------------

X = data.drop(
    "risk",
    axis=1
)

y = data["risk"]


# ------------------------------------------------------------
# 6. Split dataset into training and testing data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ------------------------------------------------------------
# 7. Create Random Forest model
# ------------------------------------------------------------

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)


# ------------------------------------------------------------
# 8. Train the AI model
# ------------------------------------------------------------

print("\nTraining AI model...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ------------------------------------------------------------
# 9. Test the model
# ------------------------------------------------------------

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n======================================")

print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)

print(
    "Risk Classes:",
    list(model.classes_)
)

print("======================================")


# ------------------------------------------------------------
# 10. Save trained model
# ------------------------------------------------------------

model_path = "models/health_risk_model.pkl"

joblib.dump(
    model,
    model_path
)


print("\nAI model saved successfully!")

print(
    f"Model location: {model_path}"
)

print("\n======================================")
print(" MODEL TRAINING COMPLETE ")
print("======================================\n")