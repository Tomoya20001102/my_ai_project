from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# 1. 学習データ（最低限）
texts = [
    "I love this movie",
    "This film is great",
    "I hate this movie",
    "This film is terrible"
]

labels = [1, 1, 0, 0]  # 1=ポジティブ, 0=ネガティブ

# 2. 単語の数値化（Bag of Words）
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 3. モデル学習
clf = LogisticRegression()
clf.fit(X, labels)

# 4. テスト
test_text = "I love this!"
test_X = vectorizer.transform([test_text])
result = clf.predict(test_X)[0]

print("Input:", test_text)
print("Prediction:", "Positive" if result == 1 else "Negative")
