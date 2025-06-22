# 🏦 Loan Approval Prediction

An end-to-end machine learning project to predict loan approvals using logistic regression.

## 📁 Project Structure

- `Scripts/`: Contains all modular scripts (data loading, preprocessing, training, prediction)
- `Models/`: Trained model & scaler (`model.pkl`, `scaler.pkl`)
- `Data/`: Input data (`train.csv`, `test.csv`)
- `main.py`: Main pipeline script for training and evaluation
- `streamlit_app.py`: Streamlit web app for interactive predictions
- `requirements.txt`: Python dependencies

## ⚙️ How to Run

1. **Clone the repo**
2. **Install dependencies**  

pip install -r requirements.txt

3. **Train and evaluate the model**  

python main.py

This will print the model accuracy and save the trained model and scaler in the `Models/` directory.

4. **Run the Streamlit app**  

streamlit run streamlit_app.py

Open the provided URL in your browser to use the loan approval predictor.

## ✅ Features

- Data preprocessing and cleaning
- Feature engineering
- Model training (Logistic Regression)
- Model evaluation (accuracy, confusion matrix)
- Model persistence (save/load)
- Interactive prediction via Streamlit web app

## 💡 Model

- **Algorithm:** Logistic Regression
- **Scaler:** StandardScaler
- **Accuracy:** Displayed in output after training

## 📊 Example Prediction

You can use the Streamlit app to input applicant details and get instant loan approval predictions.

---

**Author:**  
Nilesh Sharma