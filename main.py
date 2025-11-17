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
