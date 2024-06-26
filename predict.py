from tensorflow.keras.models import load_model
import joblib
from preprocess import preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('../model/fake_news_lstm_model.h5')
tokenizer = joblib.load('../model/tokenizer.pkl')

max_len = 200

def predict(text):
    text_cleaned = preprocess_text(text)
    text_sequence = tokenizer.texts_to_sequences([text_cleaned])
    text_padded = pad_sequences(text_sequence, maxlen=max_len)
    prediction = model.predict(text_padded)
    
    if prediction[0][0] > 0.5:
        return 'News is true'
    else:
        return 'News is fake'
