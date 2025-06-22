from Scripts.data_loader import load_data
from Scripts.preprocess import clean_data, scale_data
from Scripts.trained_model import train, evaluate, save_model, load_model
from Scripts.predict import predict_loan
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and clean data
df = load_data("data/train.csv") 
df=df.drop(columns=['Loan_ID']) # Adjust path as needed
df = clean_data(df)

# Split features and target
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
Xtrain_scaled, Xtest_scaled, scaler = scale_data(x_train, x_test)


# Train and evaluate
model = train(Xtrain_scaled, y_train)
accuracy = evaluate(model, Xtest_scaled, y_test)

print("Model accuracy is", round(accuracy, 4))

# Save model and scaler
save_model(model, scaler)

# Sample prediction
sample_input = [1, 1, 0, 1, 0, 10000, 0, 1000, 360, 1.0, 2]


prediction_sample = predict_loan(model, scaler, [sample_input])
print(prediction_sample[0])

print("Probabilities:", model.predict_proba(scaler.transform([sample_input])))