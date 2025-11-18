# ===== データセット =====
dataset = [
    ["I am happy today", 1],
    ["This is a great day", 1],
    ["I feel amazing", 1],
    ["I am sad", 0],
    ["This is terrible", 0],
    ["I feel bad", 0]
]

# ===== ポジネガ単語辞書 =====
positive_words = ["happy", "great", "amazing", "good", "nice", "love"]
negative_words = ["sad", "terrible", "bad", "hate", "angry", "awful"]

# ===== 単語数 =====
def count_words(sentence):
    return len(sentence.split())

# ===== ポジネガ単語数 =====
def count_sentiment_words(sentence):
    words = sentence.lower().split()
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    return pos, neg

# ===== 特徴量 X と 正解 y =====
X = []
y = []
for text, label in dataset:
    wc = count_words(text)
    pos, neg = count_sentiment_words(text)
    X.append([wc, pos, neg])
    y.append(label)

# ===== ルールベースAI =====
def rule_based_model(features):
    wc, pos, neg = features

    if pos > neg:
        return 1
    if neg > pos:
        return 0

    return 1 if wc > 4 else 0

def accuracy_rule_based(X, y):
    correct = 0
    for features, label in zip(X, y):
        if rule_based_model(features) == label:
            correct += 1
    return correct / len(y)

# ===== 評価 =====
acc = accuracy_rule_based(X, y)
print("ルールベースAIの正解率：", acc)
