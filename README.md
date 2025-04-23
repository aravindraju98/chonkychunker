# 🧠 chonkychunker

**chonkychunker** is a lightweight Python library for semantically clustering related text chunks using `SentenceTransformers` and `BallTree` or `NearestNeighbors`. It's ideal for preparing data for vector databases, semantic search, and LangChain RAG pipelines.

---

## 🔍 Features

- ⚡ Fast Ball Tree / Nearest Neighbors-based clustering
- 🧬 Sentence-BERT embeddings for rich semantic context
- 🧠 Cosine or Euclidean distance support
- 🔁 Removes overlap and deduplicates clusters
- 📦 Outputs ready for vector DBs (FAISS, Qdrant, Pinecone, etc.)
- 🔗 Easily converts to LangChain `Document` format

---

## 📦 Installation

```bash
pip install chonkychunker
```

Or install from source:

```bash
git clone https://github.com/aravindraju98/chonkychunker.git
cd chonkychunker
pip install -e .
```

---

## 🚀 Quickstart

```python
from chonkychunker import TextChunker

texts = [
    "The milk is spoiled.",
    "The Egg is boiled.",
    "Salt is added for taste.",
    "Thermonuclear physics is easy.",
    "Car is washed.",
    "Water is added to cooldown.",
    "Detergent is good for removing stains."
]

chunker = TextChunker(metric='cosine', top_k=4, distance_threshold=0.4)
chunker.embed(texts)
clusters = chunker.cluster()

for i, group in enumerate(clusters):
    print(f"\nCluster {i+1}:")
    for text in group:
        print(" -", text)
```

---

## 📘 LangChain Integration

```python
docs = chunker.to_langchain_documents()

# Use with FAISS or other LangChain vector store
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding)
```

---

## 🧪 Testing

```bash
pip install -r requirements.txt
python -m unittest tests/test_chunker.py
```

---

## 📜 License

This project is licensed under the MIT License.
