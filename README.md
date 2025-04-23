# 🧠 chonkychunker

**chonkychunker** is a lightweight and customizable Python library for semantically chunking and clustering texts using `SentenceTransformers` and `BallTree` or `KNN`. It’s ideal for preparing grouped content for vector databases, semantic search systems, or integration into LangChain-based RAG pipelines.

---

## 🚀 Features

- ✨ Uses `SentenceTransformer` embeddings (`all-MiniLM-L6-v2`)
- 🧠 Clusters similar texts using Ball Tree or Nearest Neighbors (KNN)
- 🔁 Deduplicates overlapping clusters
- 🔗 Outputs clusters as:
  - List of grouped text
  - LangChain-compatible `Document` objects
  - Vector DB-friendly dicts with embeddings
- 📌 Option to **merge** texts in a cluster into a single document
- 🧱 Optional **token limit** to truncate merged content for context windows

---

## 📦 Installation

```bash
pip install chonkychunker
```

Or from source:

```bash
git clone https://github.com/aravindraju98/chonkychunker.git
cd chonkychunker
pip install -e .
```

---

## 🧪 Quickstart Example

```python
from chonkychunker import TextChunker

texts = [
    "The milk is spoiled.",
    "Eggs are boiled and tasty.",
    "Physics involves matter and energy.",
    "Salt is added for flavor.",
    "Thermonuclear reactions are powerful."
]

chunker = TextChunker(metric='cosine', top_k=4, distance_threshold=0.5, max_tokens=50)
chunker.embed(texts)

# Vector output with merged clusters
vector_data = chunker.get_vector_output(merge=True)

# LangChain Documents (merged)
docs = chunker.to_langchain_documents(merge=True)
```

---

## 🔄 Merge Option

Use `merge=True` in:
- `get_vector_output(merge=True)`
- `to_langchain_documents(merge=True)`

This will concatenate all texts in a cluster into one document. If `max_tokens` is set, it will truncate the combined text based on token count using the Sentence-BERT tokenizer.

---

## 🧠 Cosine Distance vs Euclidean

- Default distance metric: `euclidean` (used with `BallTree`)
- Set `metric='cosine'` to switch to `NearestNeighbors` (KNN)
  ```python
  TextChunker(metric='cosine', ...)
  ```

---

## 📘 LangChain Integration

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding)
```

---

## 📜 License

MIT License © 2024 [Aravind Raju](https://github.com/aravindraju98)
