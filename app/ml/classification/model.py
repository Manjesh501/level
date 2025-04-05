
import joblib
from .preprocess import clean_text

model = joblib.load("models/document_classifier.pkl")

def predict(text):
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    confidence = max(model.decision_function([cleaned])[0])  # For LinearSVC
    criterion = f"Criterion {prediction.split('-')[0].strip()}"
    metric = prediction.split('-')[1].strip()
    
    return {
        "criterion": criterion,
        "metric": metric,
        "confidence": round(confidence, 2)
    }
