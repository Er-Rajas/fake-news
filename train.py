import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import joblib

from preprocess import preprocess_text

# Load datasets
df_fake = pd.read_csv('../data/fake_news.csv')
df_real = pd.read_csv('../data/real_news.csv')

# Add label column
df_fake['label'] = 0  # Fake news
df_real['label'] = 1  # Real news

# Combine datasets
df = pd.concat([df_fake, df_real]).sample(frac=1).reset_index(drop=True)

# Preprocess text
df['clean_text'] = df['text'].apply(preprocess_text)

# Tokenize and pad sequences
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['clean_text'])

X = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(X, maxlen=max_len)

y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Save the model and tokenizer
model.save('../model/fake_news_lstm_model.h5')
joblib.dump(tokenizer, '../model/tokenizer.pkl')
