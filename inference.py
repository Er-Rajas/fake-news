import joblib
from preprocessing import preprocess_text

# Load model and vectorizer
model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

def predict(text):
    text_cleaned = preprocess_text(text)
    text_vectorized = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_vectorized)
    
    if prediction[0] == 'FAKE':
        return 'News is fake'
    else:
        return 'News is true'
