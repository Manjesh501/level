import joblib
import os
from app.ml.classification.preprocess import clean_text

def predict_criterion(text):
    """
    Predict NAAC criterion for given text input
    """
    # Load model artifacts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../../../models/document_classifier.pkl")
    
    try:
        model_artifacts = joblib.load(model_path)
        pipeline = model_artifacts['pipeline']
        labels_map = model_artifacts['labels_map']
        
        # Preprocess input text
        cleaned_text = clean_text(text)
        
        # Make prediction
        prediction = pipeline.predict([cleaned_text])[0]
        
        # Get criterion description
        criterion_name = labels_map[prediction]
        
        return {
            "criterion": prediction,
            "name": criterion_name
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def test_predefined_examples():
    """Test with predefined examples"""
    test_texts = [
        "The college conducts regular workshops on research methodology.",
        "Smart classrooms are equipped with digital boards and projectors.",
        "Students actively participate in cultural and sports events.",
        "The institution has implemented rainwater harvesting systems.",
    ]
    
    print("Testing NAAC Document Classifier with Predefined Examples\n")
    for text in test_texts:
        print("\nInput Text:", text)
        result = predict_criterion(text)
        if "error" in result:
            print("Error:", result["error"])
        else:
            print(f"Predicted Criterion: {result['criterion']}")
            print(f"Criterion Name: {result['name']}")

def interactive_testing():
    """Interactive testing with user input"""
    print("\n=== Interactive NAAC Document Classification ===")
    print("Enter statements about your institution to classify them into NAAC criteria.")
    print("Enter 'q' to quit the program.\n")
    
    while True:
        print("\nEnter text to classify (or 'q' to quit):")
        text = input().strip()
        
        if text.lower() == 'q':
            print("\nThank you for using the NAAC Document Classifier!")
            break
            
        if not text:
            print("Please enter some text to classify.")
            continue
            
        result = predict_criterion(text)
        if "error" in result:
            print("Error:", result["error"])
        else:
            print(f"\nPredicted Criterion: {result['criterion']}")
            print(f"Criterion Name: {result['name']}")

if __name__ == "__main__":
    test_predefined_examples()
    interactive_testing()