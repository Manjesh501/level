import nlpaug.augmenter.word as naw
import nltk
from nltk.corpus import wordnet
import numpy as np
import pandas as pd

def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if len(word) > 3]))
    n = min(n, len(random_word_list))
    
    for _ in range(n):
        random_word = np.random.choice(random_word_list)
        synonyms = []
        for syn in wordnet.synsets(random_word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if synonyms:
            synonym = np.random.choice(list(set(synonyms)))
            new_words = [synonym if word == random_word else word for word in new_words]
    
    return ' '.join(new_words)

def augment_minority_classes(df, min_samples=5):
    """Augment classes that have fewer than min_samples"""
    augmented_data = []
    class_counts = df['criterion_label'].value_counts()
    
    for label in class_counts.index:
        count = class_counts[label]
        if count < min_samples:
            class_df = df[df['criterion_label'] == label]
            samples_needed = min_samples - count
            
            for _ in range(samples_needed):
                sample = class_df.sample(n=1).iloc[0]
                new_text = synonym_replacement(sample['text'])
                augmented_data.append({
                    'text': new_text,
                    'label_criterion': sample['label_criterion'],
                    'label_metric': sample['label_metric'],
                    'criterion_label': sample['criterion_label']
                })
    
    if augmented_data:
        augmented_df = pd.DataFrame(augmented_data)
        return pd.concat([df, augmented_df], ignore_index=True)
    return df