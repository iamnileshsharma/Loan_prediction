from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train(x_train, y_train):
   

    model = LogisticRegression(class_weight='balanced',max_iter=1000)
    model.fit(x_train, y_train)

    return model

def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test,y_pred))
    return accuracy


def save_model(model, scaler, model_path='Models/model.pkl', scaler_path='Models/scaler.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model(model_path='Models/model.pkl', scaler_path='Models/scaler.pkl'):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
