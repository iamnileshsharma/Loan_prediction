def predict_loan(model, scaler, data):
    transformed=scaler.transform(data)
    return model.predict(transformed)
