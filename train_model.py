import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from preprocessing import preprocess_text

# Ensure model directory exists
if not os.path.exists('model'):
    os.makedirs('model')

# Load datasets
df_fake = pd.read_csv('News _dataset\Fake.csv')
df_real = pd.read_csv('News _dataset\True.csv')

# Add label column
df_fake['label'] = 'FAKE'
df_real['label'] = 'REAL'

# Combine datasets
df = pd.concat([df_fake, df_real]).sample(frac=1).reset_index(drop=True)

# Preprocess text
df['clean_text'] = df['text'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='FAKE')
recall = recall_score(y_test, y_pred, pos_label='FAKE')
f1 = f1_score(y_test, y_pred, pos_label='FAKE')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Save model and vectorizer
joblib.dump(model, 'model/fake_news_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')
