import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from representations.word_embedder import WordEmbedder


if __name__ == "__main__":
    embedder = WordEmbedder("glove-wiki-gigaword-50")

    # === Test 1: Vector của một từ ===
    word = "king"
    vec = embedder.get_vector(word)
    print(f"🔹 Vector của '{word}':\n", vec[:10], "...")  # in 10 phần tử đầu

    # === Test 2: Similarity ===
    sim_king_queen = embedder.get_similarity("king", "queen")
    sim_king_man = embedder.get_similarity("king", "man")
    print(f"\n👑 Similarity(king, queen) = {sim_king_queen:.4f}")
    print(f"🧍 Similarity(king, man)   = {sim_king_man:.4f}")

    # === Test 3: Most similar ===
    print("\n🧠 Các từ giống 'computer' nhất:")
    for w, score in embedder.get_most_similar("computer", top_n=5):
        print(f"  {w:<15} {score:.4f}")

    # === Test 4: Document embedding ===
    doc = "Artificial intelligence and machine learning are closely related fields"
    vec_doc = embedder.embed_document(doc)
    print(f"\n📄 Embedding của văn bản (10 phần tử đầu): {vec_doc[:10]}")
