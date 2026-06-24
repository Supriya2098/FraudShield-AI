
import os

print("=" * 60)
print("🛡️ FRAUDSHIELD AI")
print("=" * 60)

required_files = [
    "models/random_forest.pkl",
    "reports/fraud_summary.csv",
    "reports/fraud_predictions.csv",
    "reports/top_fraud_cases.csv",
    "reports/shap_summary_fixed.png",
    "reports/shap_bar_fixed.png"
]

print("\n📂 Checking Project Files\n")

for file in required_files:
    if os.path.exists(file):
        print(f"{file} ✅")
    else:
        print(f"{file} ❌")

print("\n📊 Project Modules")
print("✓ Data Cleaning")
print("✓ Feature Engineering")
print("✓ Random Forest Model")
print("✓ Fraud Prediction")
print("✓ SHAP Explainability")
print("✓ Fraud Reports")
print("✓ Streamlit Dashboard")

print("\n🚀 Launch Dashboard:")
print("streamlit run dashboard/app.py")

print("\n🎉 FraudShield AI Ready!")
print("=" * 60)
