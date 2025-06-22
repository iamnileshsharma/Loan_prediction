import pandas as pd
import joblib
test_df = pd.read_csv("data/test.csv")  # Adjust path if needed
original_ids = test_df["Loan_ID"].copy()  # Save IDs for output
test_df.drop(columns=["Loan_ID"], inplace=True)  # Drop for prediction
from Scripts.preprocess import clean_data
test_df = clean_data(test_df)


scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/model.pkl")
test_scaled = scaler.transform(test_df)
predictions = model.predict(test_scaled)
# Convert 1 → 'Y', 0 → 'N'
predicted_labels = ['Y' if pred == 1 else 'N' for pred in predictions]

# Create output DataFrame
result_df = pd.DataFrame({
    "Loan_ID": original_ids,
    "Loan_Status": predicted_labels
})

# Save to CSV
result_df.to_csv("submission.csv", index=False)
print("✅ Predictions saved to submission.csv")
