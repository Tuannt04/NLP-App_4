## Lab 4: Word Embeddings

### Mục tiêu
- Hiểu khái niệm **Word Embeddings** – biểu diễn từ vựng dưới dạng vector số học.  
- Thực hành huấn luyện mô hình **Word2Vec** với dữ liệu thực tế.  
- So sánh kết quả giữa mô hình **tự huấn luyện** và **mô hình pre-trained**.

---

### Giới thiệu
Word Embedding là kỹ thuật ánh xạ các từ trong ngôn ngữ tự nhiên thành các vector số có ý nghĩa ngữ nghĩa.  
Các từ có ý nghĩa tương tự sẽ có vector gần nhau trong không gian vector.  
Một số mô hình phổ biến: **Word2Vec**, **GloVe**, **FastText**.

---

### Dữ liệu sử dụng
- Tập dữ liệu huấn luyện: Trích từ **C4 corpus** hoặc **Universal Dependencies (UD English-EWT)**.  
- Dữ liệu ở dạng câu, được tách từ (tokenized) trước khi đưa vào mô hình.

---

### Huấn luyện mô hình Word2Vec
Mô hình Word2Vec gồm hai kiến trúc chính:
- **CBOW (Continuous Bag of Words):** dự đoán từ trung tâm dựa vào ngữ cảnh xung quanh.  
- **Skip-gram:** dự đoán ngữ cảnh dựa vào từ trung tâm.

Trong thí nghiệm này, ta huấn luyện mô hình Skip-gram bằng thư viện **Gensim**.

---

### Mô hình Word2Vec Pre-trained
Ngoài việc tự huấn luyện, ta cũng có thể sử dụng mô hình đã huấn luyện sẵn trên tập dữ liệu lớn, ví dụ:
- `word2vec-google-news-300`  
- `glove-wiki-gigaword-100`  
Các mô hình này thường cho kết quả tốt hơn khi dữ liệu huấn luyện nhỏ.

---

### 5️⃣ Phân tích kết quả

#### Nhận xét về độ tương đồng và các từ đồng nghĩa tìm được từ model pre-trained
- Mức độ tương đồng giữa **king** và **queen** cao hơn đáng kể so với **king** và **man**.  
  Điều này hợp lý vì *king* và *queen* có mối quan hệ ngữ nghĩa gần gũi (vua – hoàng hậu),  
  trong khi *king* và *man* chỉ cùng thuộc giới tính nam.  
- Khi tìm các từ gần nhất với **computer**, model pre-trained trả về các từ như  
  *technology*, *software*, *internet*.  
  Điều này cho thấy mô hình học được không gian ngữ nghĩa rộng, liên kết *computer* với  
  toàn bộ lĩnh vực công nghệ thông tin.

---

#### Phân tích biểu đồ trực quan hóa
- Cả hai biểu đồ **PCA** và **t-SNE** đều cho thấy các từ được nhóm lại theo ngữ nghĩa như kỳ vọng.  
- **Cụm Quốc gia – Thủ đô:** *paris* gần *france*, *berlin* gần *germany*, chứng tỏ model học được  
  mối quan hệ “là thủ đô của”.  
- **Cụm Công ty Công nghệ:** *apple*, *google*, *microsoft* tạo thành cụm rõ ràng trong biểu đồ t-SNE.  
  Đây là dấu hiệu tốt vì chúng cùng thuộc lĩnh vực công nghệ.  
- **So sánh PCA và t-SNE:**  
  - PCA giữ cấu trúc tổng thể tốt, nhưng khó tách biệt các nhóm nhỏ.  
  - t-SNE thể hiện rõ ràng hơn ranh giới giữa các cụm, giúp dễ quan sát mối quan hệ ngữ nghĩa.

---

#### So sánh giữa model pre-trained và model tự huấn luyện
- **Model Pre-trained (GloVe):**  
  - Có phạm vi kiến thức rộng và tổng quát.  
  - Tìm ra các từ đồng nghĩa phổ biến: *computer → software, technology*.  
  - Phù hợp với các tác vụ NLP mang tính phổ quát.

- **Model tự huấn luyện (Spark trên C4):**  
  - Học được các mối quan hệ cụ thể và chi tiết hơn: *computer → desktop, laptop, linux*.  
  - Thể hiện sắc thái ngữ nghĩa đặc thù của tập dữ liệu web C4.  
  - Phù hợp với các ứng dụng cần ngữ nghĩa gắn với ngữ cảnh cụ thể.

---

### Đánh giá mô hình
Các cách đánh giá thông dụng:
- **Kiểm tra tương tự ngữ nghĩa:** tìm các từ gần nhất trong không gian vector.  
- **Kiểm tra quan hệ ngữ nghĩa:** ví dụ phép toán vector:  
  \`\`\`
  king - man + woman ≈ queen
  \`\`\`
  *(để minh họa, không chạy lệnh này trong Notebook)*  
- **Độ tương quan với tập chuẩn** (WordSim-353, MEN dataset…).

---

### Ứng dụng
- Phân loại văn bản (Text Classification).  
- Phân tích cảm xúc (Sentiment Analysis).  
- Hệ thống gợi ý (Recommendation).  
- Tìm kiếm ngữ nghĩa (Semantic Search).  
- Làm đầu vào cho các mô hình nâng cao hơn như RNN, LSTM, Transformer.

