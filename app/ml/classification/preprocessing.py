import pandas as pd
import json
import os
def prepare_dataset():
    # Read raw data
    df = pd.read_csv("C:/Users/TIWAR/Desktop/level-up/ai_inspection_backend/data/naac_documents.csv")
    
        # Convert numeric columns to string before concatenation
    df['label_criterion'] = df['label_criterion'].astype(str)
    df['label_metric'] = df['label_metric'].astype(str)
    
    # Create labels
    df['label'] = df['label_criterion'] + ' - ' + df['label_metric']
    df['criterion_label'] = 'Criterion ' + df['label_criterion']
    
    # Get unique criteria
    criteria = df['criterion_label'].unique()
    
    # Create labels dictionary
    labels = {}
    for criterion in sorted(criteria):
        num = criterion.split()[1]
        name = get_criterion_name(num)
        labels[criterion] = name
    
    # Save labels with proper path
    labels_path = os.path.join(os.path.dirname(__file__), "C:/Users/TIWAR/Desktop/level-up/ai_inspection_backend/data/labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=4)
    
    print("Dataset prepared successfully!")
    print(f"Total samples: {len(df)}")
    print("\nSamples per criterion:")
    print(df['criterion_label'].value_counts())
    
    # Save processed dataset
    processed_path = os.path.join(os.path.dirname(__file__), "../../data/processed/naac_documents_processed.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\nProcessed dataset saved to: {processed_path}")

def get_criterion_name(num):
    names = {
        "1": "Curricular Aspects",
        "2": "Teaching-Learning and Evaluation", 
        "3": "Research, Innovations and Extension",
        "4": "Infrastructure and Learning Resources",
        "5": "Student Support and Progression",
        "6": "Governance, Leadership and Management",
        "7": "Institutional Values and Best Practices"
    }
    return names.get(num, "Unknown")

if __name__ == "__main__":
    prepare_dataset()