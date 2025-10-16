from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from pathlib import Path

data_path = Path("data/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.txt")
save_path = Path("results/word2vec_ewt.model")

# === Đọc dữ liệu ===
print(f"🔹 Đang đọc dữ liệu từ {data_path}")
sentences = []
with open(data_path, "r", encoding="utf8") as f:
    for line in f:
        if line.strip():
            tokens = simple_preprocess(line)
            sentences.append(tokens)
print(f"✅ Đã đọc {len(sentences)} câu.")

# === Huấn luyện Word2Vec (Skip-gram hoặc CBOW) ===
model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=3,
    sg=1,          # sg=1 => Skip-gram, sg=0 => CBOW
    workers=4,
    epochs=10
)

# === Lưu mô hình ===
model.save(str(save_path))
print(f"💾 Mô hình đã lưu tại: {save_path}")

# === Kiểm tra nhanh ===
print(model.wv.most_similar("language", topn=5))
