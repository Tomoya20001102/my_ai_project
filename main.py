# ===== データセット（文章 + ラベル） =====
# 例："文章", ラベル
# ラベル 1 → ポジティブ
# ラベル 0 → ネガティブ

dataset = [
    ["I am happy today", 1],
    ["This is a great day", 1],
    ["I feel amazing", 1],
    ["I am sad", 0],
    ["This is terrible", 0],
    ["I feel bad", 0]
]

# ===== 単語数を数える関数 =====
def count_words(sentence):
    words = sentence.split()
    return len(words)

# ===== データを X（入力）と y（正解）に分ける =====
X = []
y = []

for text, label in dataset:
    word_count = count_words(text)
    X.append(word_count)
    y.append(label)

# ===== 単語AI（閾値で分類するモデル） =====
threshold = 4  # この境界より多いとポジティブ

def simple_model(word_count):
    if word_count > threshold:
        return 1  # ポジティブ
    else:
        return 0  # ネガティブ

# ===== 予測テスト =====
test_sentence = "I feel very happy today"
test_count = count_words(test_sentence)
prediction = simple_model(test_count)

print("単語数：", test_count)
print("判定結果：", prediction)

# ===== 正解率を計算する関数 =====
def accuracy(X, y, threshold):
    correct = 0
    total = len(X)

    for word_count, label in zip(X, y):
        prediction = 1 if word_count > threshold else 0
        if prediction == label:
            correct += 1
    
    return correct / total

acc = accuracy(X, y, threshold)
print("正解率：", acc)

# ===== 最適な threshold を探す =====
best_threshold = None
best_accuracy = -1

for t in range(1, 10):  # 1〜9語を試す
    acc = accuracy(X, y, t)
    print(f"threshold={t}, accuracy={acc}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = t

print("====== 結果 ======")
print("最適threshold：", best_threshold)
print("最高accuracy：", best_accuracy)
