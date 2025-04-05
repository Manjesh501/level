from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def clean_text(text):
    """Clean and preprocess the input text"""
    # Custom stopwords for NAAC domain
    custom_stops = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
        'has', 'have', 'had', 'is', 'are', 'was', 'were'
    }
    stop_words = set(stopwords.words('english')) - {'no', 'not'} # Keep negations
    stop_words.update(custom_stops)
    
    lemmatizer = WordNetLemmatizer()
    
    # Lower case
    text = text.lower()
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z\s%]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Lemmatization with POS tagging
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)