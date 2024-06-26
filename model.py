# import pandas as pd 

# df_fake=pd.read_csv("News _dataset\Fake.csv")
# df_true = pd.read_csv("News _dataset/True.csv")

# df_fake['label']="FAKE"
# df_true['label']="TRUE"

# df = pd.concat([df_fake,df_true])

# df = df.sample(frac=1).reset_index(drop=True)

# import nltk
# from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# def preprocess_text(text):
#     text = text.lower()
#     text = ''.join([c for c in text if c not in ('!', '.', ':', ';', '?', '-', '_', '(', ')', '[', ']', '{', '}', '\'')])
#     text = ' '.join([word for word in text.split() if word not in stop_words])
#     return text

# df['clean_text'] = df['text'].apply(preprocess_text)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)

# # Vectorize the text
# vectorizer = TfidfVectorizer(max_features=5000)
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# model = LogisticRegression()
# model.fit(X_train_vec, y_train)

# y_pred = model.predict(X_test_vec)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, pos_label='FAKE')
# recall = recall_score(y_test, y_pred, pos_label='FAKE')
# f1 = f1_score(y_test, y_pred, pos_label='FAKE')

# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'F1 Score: {f1}')
