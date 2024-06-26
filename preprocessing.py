import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in ('!', '.', ':', ';', '?', '-', '_', '(', ')', '[', ']', '{', '}', '\'')])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
